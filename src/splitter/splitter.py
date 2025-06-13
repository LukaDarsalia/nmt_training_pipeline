"""
Data Splitter with YAML configuration support.
"""

import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import wandb
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .contamination import ContaminationDetector

tqdm.pandas()


class DataSplitter:
    """
    A class for splitting datasets into train/valid/test sets based on YAML configuration,
    with contamination checking and comprehensive logging to wandb.
    """

    def __init__(self,
                 artifact: wandb.Artifact,
                 output_folder_dir: str,
                 config_path: str = "config/splitting.yaml"):
        """
        Initialize the DataSplitter.

        Args:
            artifact: The wandb artifact to log data to
            output_folder_dir: Directory to save split datasets
            config_path: Path to the YAML configuration file
        """
        self.artifact = artifact
        self.output_folder_dir = Path(output_folder_dir)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.run = wandb.run

        # Create output directory
        self.output_folder_dir.mkdir(parents=True, exist_ok=True)

        # Log the config file to wandb
        self._log_config_to_wandb()

        # Initialize contamination detector
        self.contamination_detector = ContaminationDetector(
            self.config.get('contamination_check', {})
        )

        # Statistics tracking
        self.split_stats = {}
        self.contamination_results = {}
        self.artifact_cache = {}  # Cache loaded artifacts

        # Set random seeds
        random_seed = self.config.get('settings', {}).get('random_seed', 42)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _log_config_to_wandb(self) -> None:
        """Log configuration and splitter files to wandb."""
        # Log config file
        self.artifact.add_file(str(self.config_path), name="splitting_config.yaml")

        # Log splitter files
        splitter_files = [
            "src/splitter/splitter.py",
            "src/splitter/contamination.py",
        ]

        for file_path in splitter_files:
            if Path(file_path).exists():
                self.artifact.add_file(str(file_path), name=f"splitter_code/{Path(file_path).name}")

    def _download_artifact(self, artifact_name: str) -> Path:
        """
        Download wandb artifact and return path to data directory.

        Args:
            artifact_name: Name of the artifact (e.g., "raw:v1", "preprocessed:latest")

        Returns:
            Path to downloaded artifact directory
        """
        if artifact_name in self.artifact_cache:
            return self.artifact_cache[artifact_name]

        print(f"Downloading artifact: {artifact_name}")
        artifact = self.run.use_artifact(artifact_name)
        artifact_dir = Path(artifact.download())

        self.artifact_cache[artifact_name] = artifact_dir
        return artifact_dir

    def _load_artifact_data(self, artifact_name: str) -> pd.DataFrame:
        """
        Load all parquet files from an artifact.

        Args:
            artifact_name: Name of the artifact

        Returns:
            Combined DataFrame from all parquet files
        """
        artifact_dir = self._download_artifact(artifact_name)

        data_frames = []
        for parquet_file in artifact_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            df['source_artifact'] = artifact_name
            df['source_file'] = parquet_file.stem
            data_frames.append(df)

        if not data_frames:
            raise ValueError(f"No parquet files found in artifact: {artifact_name}")

        combined_df = pd.concat(data_frames, ignore_index=True)
        print(f"  Loaded {len(combined_df)} rows from {artifact_name}")

        return combined_df

    def _sample_from_artifact(self,
                            artifact_name: str,
                            percentage: float,
                            seed: Optional[int] = None) -> pd.DataFrame:
        """
        Sample a percentage of data from an artifact.

        Args:
            artifact_name: Name of the artifact
            percentage: Percentage to sample (0-100)
            seed: Random seed for sampling

        Returns:
            Sampled DataFrame
        """
        df = self._load_artifact_data(artifact_name)

        if percentage >= 100:
            return df

        n_samples = int(len(df) * percentage / 100)

        if seed is not None:
            return df.sample(n=n_samples, random_state=seed)
        else:
            return df.sample(n=n_samples)

    def _stratified_sample(self,
                         df: pd.DataFrame,
                         percentage: float,
                         stratify_column: str = 'domain',
                         seed: Optional[int] = None) -> pd.DataFrame:
        """
        Perform stratified sampling to maintain domain distribution.

        Args:
            df: Input DataFrame
            percentage: Percentage to sample (0-100)
            stratify_column: Column to stratify by
            seed: Random seed

        Returns:
            Stratified sample DataFrame
        """
        if percentage >= 100 or stratify_column not in df.columns:
            return df

        # Get minimum samples per domain
        min_samples = self.config.get('settings', {}).get('min_samples_per_domain', 10)

        sampled_dfs = []
        for domain in df[stratify_column].unique():
            domain_df = df[df[stratify_column] == domain]
            n_samples = max(
                min_samples,
                int(len(domain_df) * percentage / 100)
            )
            n_samples = min(n_samples, len(domain_df))

            if n_samples > 0:
                sampled = domain_df.sample(n=n_samples, random_state=seed)
                sampled_dfs.append(sampled)

        return pd.concat(sampled_dfs, ignore_index=True)

    def _create_split(self, split_name: str, split_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a single split (train/valid/test) based on configuration.

        Args:
            split_name: Name of the split (train/valid/test)
            split_config: Configuration for this split

        Returns:
            DataFrame for this split
        """
        print(f"\nCreating {split_name} split...")

        split_dfs = []

        for source in split_config.get('sources', []):
            artifact_name = source['artifact']
            percentage = source['percentage']
            seed = source.get('seed', None)

            print(f"  Loading {percentage}% from {artifact_name}")

            # Check if we should stratify
            stratify = self.config.get('settings', {}).get('stratify_by_domain', True)

            if stratify:
                sampled_df = self._stratified_sample(
                    self._load_artifact_data(artifact_name),
                    percentage,
                    seed=seed
                )
            else:
                sampled_df = self._sample_from_artifact(
                    artifact_name,
                    percentage,
                    seed=seed
                )

            split_dfs.append(sampled_df)

        # Combine all sources for this split
        combined_df = pd.concat(split_dfs, ignore_index=True)

        # Shuffle if configured
        if self.config.get('settings', {}).get('shuffle_before_split', True):
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"  Created {split_name} split with {len(combined_df)} samples")

        return combined_df

    def _check_contamination(self,
                           splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Check and remove contamination between splits.

        Args:
            splits: Dictionary of split DataFrames

        Returns:
            Dictionary of cleaned split DataFrames
        """
        contamination_config = self.config.get('contamination_check', {})

        if not contamination_config.get('enabled', True):
            print("\nContamination checking disabled")
            return splits

        print("\n" + "="*60)
        print("CONTAMINATION CHECKING")
        print("="*60)

        cleaned_splits = splits.copy()

        for check_pair in contamination_config.get('check_pairs', []):
            source = check_pair['source']
            target = check_pair['target']
            action = check_pair.get('action', 'remove_from_target')

            if source not in splits or target not in splits:
                print(f"Warning: Missing split for contamination check: {source} → {target}")
                continue

            # Check contamination
            clean_df, contaminated_df, stats = self.contamination_detector.check_contamination(
                splits[source],
                cleaned_splits[target],
                source,
                target
            )

            # Store results
            check_key = f"{source}_to_{target}"
            self.contamination_results[check_key] = stats

            # Log contaminated samples
            if len(contaminated_df) > 0:
                sample_size = min(20, len(contaminated_df))
                contaminated_sample = contaminated_df.sample(sample_size, random_state=42)

                # Select relevant columns for logging
                log_columns = ['en', 'ka', 'domain', 'id', 'source_artifact']
                available_columns = [col for col in log_columns if col in contaminated_sample.columns]

                table = wandb.Table(dataframe=contaminated_sample[available_columns])
                self.artifact.add(table, f"contaminated_{source}_to_{target}")

            # Apply action
            if action == 'remove_from_target' and len(contaminated_df) > 0:
                cleaned_splits[target] = clean_df
                print(f"  Removed {len(contaminated_df)} contaminated samples from {target}")
            elif action == 'warn_only':
                print(f"  WARNING: Found {len(contaminated_df)} contaminated samples in {target}")

        return cleaned_splits

    def _log_split_statistics(self, splits: Dict[str, pd.DataFrame]) -> None:
        """Log comprehensive statistics about the splits."""
        print("\n" + "="*60)
        print("SPLIT STATISTICS")
        print("="*60)

        # Overall statistics
        total_samples = sum(len(df) for df in splits.values())

        for split_name, split_df in splits.items():
            split_size = len(split_df)
            split_percentage = (split_size / total_samples * 100) if total_samples > 0 else 0

            stats = {
                'size': split_size,
                'percentage': split_percentage,
                'unique_sources': split_df['source_artifact'].nunique() if 'source_artifact' in split_df.columns else 0
            }

            # Domain distribution if available
            if 'domain' in split_df.columns:
                domain_counts = split_df['domain'].value_counts()
                stats['num_domains'] = len(domain_counts)
                stats['domain_distribution'] = domain_counts.to_dict()

            self.split_stats[split_name] = stats

            print(f"\n{split_name.upper()} Split:")
            print(f"  Size: {split_size:,} ({split_percentage:.1f}%)")
            print(f"  Sources: {stats['unique_sources']}")

            if 'domain' in split_df.columns:
                print(f"  Domains: {stats['num_domains']}")
                print("  Domain distribution:")
                for domain, count in domain_counts.head(10).items():
                    print(f"    {domain}: {count:,} ({count/split_size*100:.1f}%)")

        # Log to wandb
        wandb.log({
            'total_samples': total_samples,
            'train_size': self.split_stats.get('train', {}).get('size', 0),
            'valid_size': self.split_stats.get('valid', {}).get('size', 0),
            'test_size': self.split_stats.get('test', {}).get('size', 0),
        })

        # Create distribution plot if configured
        if self.config.get('settings', {}).get('log_distributions', True):
            self._log_distribution_plots(splits)

    def _log_distribution_plots(self, splits: Dict[str, pd.DataFrame]) -> None:
        """Log distribution plots to wandb."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Plot 1: Split sizes
        split_sizes = [len(df) for df in splits.values()]
        split_names = list(splits.keys())

        axes[0].bar(split_names, split_sizes)
        axes[0].set_title('Split Sizes')
        axes[0].set_ylabel('Number of Samples')

        # Add value labels on bars
        for i, (name, size) in enumerate(zip(split_names, split_sizes)):
            axes[0].text(i, size + max(split_sizes)*0.01, f'{size:,}', ha='center')

        # Plot 2: Domain distribution across splits (if available)
        if all('domain' in df.columns for df in splits.values()):
            domain_data = []
            for split_name, split_df in splits.items():
                for domain, count in split_df['domain'].value_counts().items():
                    domain_data.append({
                        'split': split_name,
                        'domain': domain,
                        'count': count
                    })

            domain_df = pd.DataFrame(domain_data)

            # Get top domains by total count
            top_domains = (domain_df.groupby('domain')['count']
                          .sum()
                          .nlargest(10)
                          .index
                          .tolist())

            # Filter to top domains
            domain_df_filtered = domain_df[domain_df['domain'].isin(top_domains)]

            # Create grouped bar plot
            domain_pivot = domain_df_filtered.pivot(index='domain', columns='split', values='count').fillna(0)
            domain_pivot.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Top 10 Domains Across Splits')
            axes[1].set_ylabel('Number of Samples')
            axes[1].tick_params(axis='x', rotation=45)

        # Plot 3: Source artifact distribution
        if all('source_artifact' in df.columns for df in splits.values()):
            source_data = []
            for split_name, split_df in splits.items():
                for source, count in split_df['source_artifact'].value_counts().items():
                    source_data.append({
                        'split': split_name,
                        'source': source,
                        'count': count
                    })

            source_df = pd.DataFrame(source_data)
            source_pivot = source_df.pivot(index='source', columns='split', values='count').fillna(0)
            source_pivot.plot(kind='bar', ax=axes[2])
            axes[2].set_title('Source Artifacts Across Splits')
            axes[2].set_ylabel('Number of Samples')
            axes[2].tick_params(axis='x', rotation=45)

        # Plot 4: Contamination summary (if available)
        if self.contamination_results:
            contamination_data = []
            for check_pair, stats in self.contamination_results.items():
                source, target = check_pair.split('_to_')
                contamination_data.append({
                    'check': f"{source}→{target}",
                    'contaminated': stats['total_contaminated'],
                    'clean': stats['target_size'] - stats['total_contaminated']
                })

            cont_df = pd.DataFrame(contamination_data)

            # Stacked bar plot
            cont_df.set_index('check')[['clean', 'contaminated']].plot(
                kind='bar', stacked=True, ax=axes[3],
                color=['green', 'red']
            )
            axes[3].set_title('Contamination Check Results')
            axes[3].set_ylabel('Number of Samples')
            axes[3].tick_params(axis='x', rotation=45)
            axes[3].legend(['Clean', 'Contaminated'])

        plt.tight_layout()
        wandb.log({"split_distributions": wandb.Image(fig)})
        plt.close()

    def run_splitting(self) -> None:
        """Main method to run the data splitting process."""
        print(f"Starting data splitting with config: {self.config_path}")
        print(f"Output directory: {self.output_folder_dir}")

        # Create splits
        splits = {}
        for split_name in ['train', 'valid', 'test']:
            if split_name in self.config.get('splits', {}):
                split_config = self.config['splits'][split_name]
                splits[split_name] = self._create_split(split_name, split_config)
            else:
                print(f"Warning: No configuration found for {split_name} split")

        # Check contamination
        if self.config.get('contamination_check', {}).get('enabled', True):
            splits = self._check_contamination(splits)

            # Log contamination report
            if self.config.get('settings', {}).get('log_contamination_report', True):
                report_df = self.contamination_detector.generate_contamination_report(
                    self.contamination_results
                )
                if len(report_df) > 0:
                    table = wandb.Table(dataframe=report_df)
                    self.artifact.add(table, "contamination_report")

        # Save splits
        print("\nSaving splits...")
        for split_name, split_df in splits.items():
            output_path = self.output_folder_dir / f"{split_name}.parquet"
            split_df.to_parquet(output_path, index=False)
            print(f"  Saved {split_name} split: {output_path}")

            # Log sample to wandb
            sample_size = min(50, len(split_df))
            if len(split_df) > 0:
                sample = split_df.sample(sample_size, random_state=42)

                # Select relevant columns for logging
                log_columns = ['en', 'ka', 'domain', 'id', 'source_artifact']
                available_columns = [col for col in log_columns if col in sample.columns]

                table = wandb.Table(dataframe=sample[available_columns])
                self.artifact.add(table, f"{split_name}_sample")

        # Log statistics
        self._log_split_statistics(splits)

        # Update artifact metadata
        self.artifact.metadata.update({
            'split_stats': self.split_stats,
            'contamination_results': self.contamination_results,
            'num_splits': len(splits),
            'total_samples': sum(len(df) for df in splits.values())
        })

        print("\nData splitting completed!")

    def get_split_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about configured splits."""
        info = {}
        for split_name, split_config in self.config.get('splits', {}).items():
            info[split_name] = {
                'sources': [
                    {
                        'artifact': source['artifact'],
                        'percentage': source['percentage']
                    }
                    for source in split_config.get('sources', [])
                ]
            }
        return info
