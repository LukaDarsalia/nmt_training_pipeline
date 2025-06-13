"""
Data Cleaner with YAML configuration support.
"""

import time
from pathlib import Path
from typing import List

import wandb
import yaml

from .cleaners import *  # Import all cleaner functions to register them
from .registry import cleaner_registry

tqdm.pandas()


class DataCleaner:
    """
    A class for cleaning datasets based on YAML configuration,
    logging statistics and samples to wandb.
    """

    def __init__(self,
                 artifact: wandb.Artifact,
                 input_folder_dir: str,
                 output_folder_dir: str,
                 config_path: str = "config/cleaning.yaml"):
        """
        Initialize the DataCleaner.

        Args:
            artifact: The wandb artifact to log data to
            input_folder_dir: Directory containing input parquet files
            output_folder_dir: Directory to save cleaned datasets
            config_path: Path to the YAML configuration file
        """
        self.artifact = artifact
        self.input_folder_dir = Path(input_folder_dir)
        self.output_folder_dir = Path(output_folder_dir)
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Create output directory
        self.output_folder_dir.mkdir(parents=True, exist_ok=True)

        # Log the config file to wandb
        self._log_config_to_wandb()

        # Statistics tracking
        self.cleaning_stats = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _log_config_to_wandb(self) -> None:
        """Log configuration and cleaner files to wandb."""
        # Log config file
        self.artifact.add_file(str(self.config_path), name="cleaning_config.yaml")

        # Log cleaner files
        cleaner_files = [
            "src/cleaner/cleaner.py",
            "src/cleaner/cleaners.py",
            "src/cleaner/registry.py"
        ]

        for file_path in cleaner_files:
            if Path(file_path).exists():
                self.artifact.add_file(file_path, name=f"cleaner_code/{Path(file_path).name}")

    def _apply_single_cleaner(self,
                              df: pd.DataFrame,
                              cleaner_config: Dict[str, Any],
                              dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply a single cleaning function to the dataset.

        Args:
            df: Input DataFrame
            cleaner_config: Configuration for the cleaner
            dataset_name: Name of the dataset being cleaned

        Returns:
            Tuple of (cleaned_df, stats_dict)
        """
        if not cleaner_config.get("enabled", True):
            print(f"Skipping disabled cleaner: {cleaner_config['name']}")
            return df, {}

        cleaner_name = cleaner_config["name"]
        print(f"Applying cleaner: {cleaner_name}")

        # Get the cleaner function
        cleaner_function = cleaner_registry.get_cleaner(cleaner_name)
        if cleaner_function is None:
            raise ValueError(f"Cleaner '{cleaner_name}' not found. "
                             f"Available cleaners: {list(cleaner_registry.list_cleaners().keys())}")

        # Record start time
        start_time = time.time()
        original_size = len(df)

        try:
            # Apply the cleaner
            cleaned_df, dropped_df = cleaner_function(df, cleaner_config.get("params", {}))

            # Validate output
            cleaned_df, dropped_df = cleaner_registry.validate_cleaner_output(
                cleaned_df, dropped_df, df, cleaner_name
            )

            # Calculate statistics
            processing_time = time.time() - start_time
            cleaned_size = len(cleaned_df)
            dropped_size = len(dropped_df)
            retention_rate = cleaned_size / original_size if original_size > 0 else 0

            stats = {
                "original_size": original_size,
                "cleaned_size": cleaned_size,
                "dropped_size": dropped_size,
                "retention_rate": retention_rate,
                "processing_time": processing_time
            }

            # Log sample of dropped data if any
            if len(dropped_df) > 0:
                sample_size = min(10, len(dropped_df))
                dropped_sample = dropped_df.sample(sample_size, random_state=42)

                # Create wandb table
                table = wandb.Table(dataframe=dropped_sample)
                self.artifact.add(table, f"{dataset_name}_{cleaner_name}_dropped_sample")

            print(f"  ✓ {cleaner_name}: {original_size} → {cleaned_size} "
                  f"({retention_rate:.2%} retained, {processing_time:.1f}s)")

            return cleaned_df, stats

        except Exception as e:
            print(f"  ✗ Error in {cleaner_name}: {str(e)}")
            raise

    def _clean_single_dataset(self, dataset_path: Path) -> None:
        """
        Clean a single dataset file.

        Args:
            dataset_path: Path to the parquet file to clean
        """
        dataset_name = dataset_path.stem
        print(f"\nCleaning dataset: {dataset_name}")

        # Load the dataset
        try:
            df = pd.read_parquet(dataset_path)
            print(f"  Loaded {len(df)} rows from {dataset_path}")
        except Exception as e:
            print(f"  ✗ Error loading {dataset_path}: {e}")
            return

        # Initialize statistics for this dataset
        self.cleaning_stats[dataset_name] = {
            "original_size": len(df),
            "cleaners": {},
            "final_size": len(df),
            "total_retention_rate": 1.0
        }

        current_df = df.copy()

        # Apply each cleaner in sequence
        for cleaner_config in self.config.get("cleaners", []):
            if cleaner_config.get("enabled", True):
                current_df, cleaner_stats = self._apply_single_cleaner(
                    current_df, cleaner_config, dataset_name
                )

                # Store cleaner statistics
                self.cleaning_stats[dataset_name]["cleaners"][cleaner_config["name"]] = cleaner_stats

        # Update final statistics
        final_size = len(current_df)
        self.cleaning_stats[dataset_name]["final_size"] = final_size
        self.cleaning_stats[dataset_name]["total_retention_rate"] = (
            final_size / len(df) if len(df) > 0 else 0
        )

        # Save cleaned dataset
        output_path = self.output_folder_dir / f"{dataset_name}_cleaned.parquet"
        current_df.to_parquet(output_path, index=False)

        # Log sample of final cleaned data
        sample_size = min(20, len(current_df))
        if len(current_df) > 0:
            cleaned_sample = current_df.sample(sample_size, random_state=42)
            table = wandb.Table(dataframe=cleaned_sample)
            self.artifact.add(table, f"{dataset_name}_final_cleaned_sample")

        # Update artifact metadata
        self.artifact.metadata.update({
            f"{dataset_name}_original_size": len(df),
            f"{dataset_name}_final_size": final_size,
            f"{dataset_name}_retention_rate": self.cleaning_stats[dataset_name]["total_retention_rate"]
        })

        print(f"  ✓ Saved cleaned dataset: {output_path}")
        print(f"  📊 Final: {len(df)} → {final_size} "
              f"({self.cleaning_stats[dataset_name]['total_retention_rate']:.2%} retained)")

    def clean_all_datasets(self) -> None:
        """Clean all parquet files in the input directory."""
        # Find all parquet files
        parquet_files = list(self.input_folder_dir.glob("*.parquet"))

        if not parquet_files:
            print(f"No parquet files found in {self.input_folder_dir}")
            return

        print(f"Found {len(parquet_files)} dataset(s) to clean")
        print(f"Enabled cleaners: {[c['name'] for c in self.config.get('cleaners', []) if c.get('enabled', True)]}")

        # Clean each dataset
        for parquet_file in parquet_files:
            self._clean_single_dataset(parquet_file)

        # Log overall statistics
        self._log_cleaning_summary()

    def _log_cleaning_summary(self) -> None:
        """Log comprehensive cleaning statistics to wandb."""
        print("\n" + "=" * 60)
        print("CLEANING SUMMARY")
        print("=" * 60)

        # Create summary statistics
        total_original = sum(stats["original_size"] for stats in self.cleaning_stats.values())
        total_final = sum(stats["final_size"] for stats in self.cleaning_stats.values())
        overall_retention = total_final / total_original if total_original > 0 else 0

        summary_stats = {
            "total_datasets": len(self.cleaning_stats),
            "total_original_rows": total_original,
            "total_final_rows": total_final,
            "overall_retention_rate": overall_retention,
            "datasets": self.cleaning_stats
        }

        # Log to wandb
        self.artifact.metadata.update({
            "cleaning_summary": summary_stats
        })

        # Print summary
        print(f"Datasets processed: {len(self.cleaning_stats)}")
        print(f"Total rows: {total_original:,} → {total_final:,}")
        print(f"Overall retention rate: {overall_retention:.2%}")

        print("\nPer-dataset summary:")
        for dataset_name, stats in self.cleaning_stats.items():
            print(f"  {dataset_name}: {stats['original_size']:,} → {stats['final_size']:,} "
                  f"({stats['total_retention_rate']:.2%})")

        print("=" * 60)

    def run_cleaning(self) -> None:
        """Main method to initiate the cleaning process."""
        print(f"Starting data cleaning with config: {self.config_path}")
        print(f"Input directory: {self.input_folder_dir}")
        print(f"Output directory: {self.output_folder_dir}")

        self.clean_all_datasets()
        print("\nData cleaning completed!")

    def get_enabled_cleaners(self) -> List[str]:
        """Get list of enabled cleaner names."""
        return [
            cleaner["name"]
            for cleaner in self.config.get("cleaners", [])
            if cleaner.get("enabled", True)
        ]

    def get_cleaner_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured cleaners."""
        info = {}
        for cleaner in self.config.get("cleaners", []):
            info[cleaner["name"]] = {
                "description": cleaner.get("description", ""),
                "enabled": cleaner.get("enabled", True),
                "params": cleaner.get("params", {})
            }
        return info