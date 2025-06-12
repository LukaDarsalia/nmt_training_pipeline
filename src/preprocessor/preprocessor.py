"""
Data Preprocessor with YAML configuration support.
"""

from pathlib import Path
from typing import Dict, Any, List
import time

import pandas as pd
import wandb
import yaml
from tqdm import tqdm

from .registry import preprocessor_registry

tqdm.pandas()


class DataPreprocessor:
    """
    A class for preprocessing and augmenting datasets based on YAML configuration,
    logging statistics and samples to wandb.
    """

    def __init__(self,
                 artifact: wandb.Artifact,
                 input_folder_dir: str,
                 output_folder_dir: str,
                 config_path: str = "config/preprocessing.yaml"):
        """
        Initialize the DataPreprocessor.

        Args:
            artifact: The wandb artifact to log data to
            input_folder_dir: Directory containing input parquet files
            output_folder_dir: Directory to save processed datasets
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
        self.preprocessing_stats = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _log_config_to_wandb(self) -> None:
        """Log configuration and preprocessor files to wandb."""
        # Log config file
        self.artifact.add_file(str(self.config_path), name="preprocessing_config.yaml")

        # Log preprocessor files
        preprocessor_files = [
            "src/preprocessor/data_preprocessor.py",
            "src/preprocessor/preprocessing_functions.py",
            "src/preprocessor/augmentation_functions.py",
            "src/preprocessor/registry.py"
        ]

        for file_path in preprocessor_files:
            if Path(file_path).exists():
                self.artifact.add_file(file_path, name=f"preprocessor_code/{Path(file_path).name}")

    def _apply_single_preprocessor(self,
                                 df: pd.DataFrame,
                                 preprocessor_config: Dict[str, Any],
                                 dataset_name: str) -> pd.DataFrame:
        """
        Apply a single preprocessing function to the dataset.

        Args:
            df: Input DataFrame
            preprocessor_config: Configuration for the preprocessor
            dataset_name: Name of the dataset being processed

        Returns:
            Processed DataFrame
        """
        if not preprocessor_config.get("enabled", True):
            print(f"  Skipping disabled preprocessor: {preprocessor_config['name']}")
            return df

        preprocessor_name = preprocessor_config["name"]
        print(f"  Applying preprocessor: {preprocessor_name}")

        # Get the preprocessor function
        preprocessor_function = preprocessor_registry.get_preprocessor(preprocessor_name)
        if preprocessor_function is None:
            available_preprocessors = list(preprocessor_registry.list_preprocessors().keys())
            available_augmenters = list(preprocessor_registry.list_augmenters().keys())
            raise ValueError(f"Preprocessor '{preprocessor_name}' not found. "
                           f"Available preprocessors: {available_preprocessors}. "
                           f"Available augmenters: {available_augmenters}. "
                           f"Note: Make sure function is registered with @register_preprocessor decorator.")

        # Record start time
        start_time = time.time()
        original_size = len(df)

        try:
            # Apply the preprocessor
            processed_df = preprocessor_function(df, preprocessor_config.get("params", {}))

            # Validate output
            processed_df = preprocessor_registry.validate_preprocessor_output(
                processed_df, df, preprocessor_name
            )

            # Calculate statistics
            processing_time = time.time() - start_time

            # Log sample of processed data
            sample_size = min(5, len(processed_df))
            if len(processed_df) > 0:
                processed_sample = processed_df.sample(sample_size, random_state=42)

                # Create wandb table
                table = wandb.Table(dataframe=processed_sample)
                self.artifact.add(table, f"{dataset_name}_{preprocessor_name}_sample")

            print(f"    ✓ {preprocessor_name}: Processed {original_size} rows in {processing_time:.1f}s")

            return processed_df

        except Exception as e:
            print(f"    ✗ Error in {preprocessor_name}: {str(e)}")
            raise

    def _apply_single_augmenter(self,
                              df: pd.DataFrame,
                              augmenter_config: Dict[str, Any],
                              dataset_name: str) -> pd.DataFrame:
        """
        Apply a single augmentation function to the dataset.

        Args:
            df: Input DataFrame
            augmenter_config: Configuration for the augmenter
            dataset_name: Name of the dataset being augmented

        Returns:
            DataFrame with augmented examples
        """
        if not augmenter_config.get("enabled", True):
            print(f"  Skipping disabled augmenter: {augmenter_config['name']}")
            return pd.DataFrame()

        augmenter_name = augmenter_config["name"]
        print(f"  Applying augmenter: {augmenter_name}")

        # Get the augmenter function
        augmenter_function = preprocessor_registry.get_augmenter(augmenter_name)
        if augmenter_function is None:
            available_preprocessors = list(preprocessor_registry.list_preprocessors().keys())
            available_augmenters = list(preprocessor_registry.list_augmenters().keys())
            raise ValueError(f"Augmenter '{augmenter_name}' not found. "
                           f"Available preprocessors: {available_preprocessors}. "
                           f"Available augmenters: {available_augmenters}. "
                           f"Note: Make sure function is registered with @register_augmenter decorator.")

        # Record start time
        start_time = time.time()

        try:
            # Apply the augmenter
            augmented_df = augmenter_function(df, augmenter_config.get("params", {}))

            # Validate output
            if len(augmented_df) > 0:
                augmented_df = preprocessor_registry.validate_augmenter_output(
                    augmented_df, df, augmenter_name
                )

            # Calculate statistics
            processing_time = time.time() - start_time
            augmented_size = len(augmented_df)

            # Log sample of augmented data
            if augmented_size > 0:
                sample_size = min(10, augmented_size)
                augmented_sample = augmented_df.sample(sample_size, random_state=42)

                # Create wandb table
                table = wandb.Table(dataframe=augmented_sample)
                self.artifact.add(table, f"{dataset_name}_{augmenter_name}_sample")

            print(f"    ✓ {augmenter_name}: Generated {augmented_size} new examples in {processing_time:.1f}s")

            return augmented_df

        except Exception as e:
            print(f"    ✗ Error in {augmenter_name}: {str(e)}")
            raise

    def _process_single_dataset(self, dataset_path: Path) -> None:
        """
        Process a single dataset file.

        Args:
            dataset_path: Path to the parquet file to process
        """
        dataset_name = dataset_path.stem
        print(f"\nProcessing dataset: {dataset_name}")

        # Load the dataset
        try:
            df = pd.read_parquet(dataset_path)
            print(f"  Loaded {len(df)} rows from {dataset_path}")
        except Exception as e:
            print(f"  ✗ Error loading {dataset_path}: {e}")
            return

        # Initialize statistics for this dataset
        self.preprocessing_stats[dataset_name] = {
            "original_size": len(df),
            "preprocessors": {},
            "augmenters": {},
            "final_size": len(df),
            "augmentation_ratio": 0.0
        }

        current_df = df.copy()

        # Apply preprocessing steps
        print("  --- Preprocessing Steps ---")
        for preprocessor_config in self.config.get("preprocessors", []):
            if preprocessor_config.get("enabled", True):
                current_df = self._apply_single_preprocessor(
                    current_df, preprocessor_config, dataset_name
                )

        # Apply augmentation steps
        print("  --- Augmentation Steps ---")
        all_augmented_data = []
        dataset_restructured = False

        for augmenter_config in self.config.get("augmenters", []):
            if augmenter_config.get("enabled", True):
                augmenter_name = augmenter_config["name"]

                # Special handling for dataset restructuring operations
                if augmenter_name == "concatenate_long_texts":
                    # This replaces the dataset rather than adding to it
                    restructured_df = self._apply_single_augmenter(
                        current_df, augmenter_config, dataset_name
                    )
                    if len(restructured_df) > 0:
                        current_df = restructured_df  # Replace the dataset
                        dataset_restructured = True
                        print(f"    ✓ Dataset restructured by {augmenter_name}")
                else:
                    # Normal augmentation - adds new examples
                    augmented_df = self._apply_single_augmenter(
                        current_df, augmenter_config, dataset_name
                    )

                    if len(augmented_df) > 0:
                        all_augmented_data.append(augmented_df)

        # Combine original and augmented data (if not restructured)
        if all_augmented_data and not dataset_restructured:
            combined_augmented = pd.concat(all_augmented_data, ignore_index=True)
            final_df = pd.concat([current_df, combined_augmented], ignore_index=True)
            print(f"  Combined {len(combined_augmented)} augmented examples with {len(current_df)} original")
        elif all_augmented_data and dataset_restructured:
            # Dataset was restructured, but we still have normal augmentation
            combined_augmented = pd.concat(all_augmented_data, ignore_index=True)
            final_df = pd.concat([current_df, combined_augmented], ignore_index=True)
            print(f"  Combined {len(combined_augmented)} augmented examples with {len(current_df)} restructured")
        else:
            final_df = current_df
            if dataset_restructured:
                print("  Dataset restructured, no additional augmented examples")
            else:
                print("  No augmented examples generated")

        # Update final statistics
        final_size = len(final_df)
        augmentation_ratio = (final_size - len(current_df)) / len(current_df) if len(current_df) > 0 else 0

        self.preprocessing_stats[dataset_name]["final_size"] = final_size
        self.preprocessing_stats[dataset_name]["augmentation_ratio"] = augmentation_ratio

        # Save processed dataset
        output_path = self.output_folder_dir / f"{dataset_name}_processed.parquet"
        final_df.to_parquet(output_path, index=False)

        # Log sample of final processed data
        sample_size = min(20, len(final_df))
        if len(final_df) > 0:
            final_sample = final_df.sample(sample_size, random_state=42)
            table = wandb.Table(dataframe=final_sample)
            self.artifact.add(table, f"{dataset_name}_final_processed_sample")

        # Update artifact metadata
        self.artifact.metadata.update({
            f"{dataset_name}_original_size": len(df),
            f"{dataset_name}_final_size": final_size,
            f"{dataset_name}_augmentation_ratio": augmentation_ratio
        })

        print(f"  ✓ Saved processed dataset: {output_path}")
        print(f"  📊 Final: {len(df)} → {final_size} "
              f"({augmentation_ratio:.2%} augmentation)")

    def process_all_datasets(self) -> None:
        """Process all parquet files in the input directory."""
        # Find all parquet files
        parquet_files = list(self.input_folder_dir.glob("*.parquet"))

        if not parquet_files:
            print(f"No parquet files found in {self.input_folder_dir}")
            return

        print(f"Found {len(parquet_files)} dataset(s) to process")
        print(f"Enabled preprocessors: {[p['name'] for p in self.config.get('preprocessors', []) if p.get('enabled', True)]}")
        print(f"Enabled augmenters: {[a['name'] for a in self.config.get('augmenters', []) if a.get('enabled', True)]}")

        # Process each dataset
        for parquet_file in parquet_files:
            self._process_single_dataset(parquet_file)

        # Log overall statistics
        self._log_processing_summary()

    def _log_processing_summary(self) -> None:
        """Log comprehensive processing statistics to wandb."""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)

        # Create summary statistics
        total_original = sum(stats["original_size"] for stats in self.preprocessing_stats.values())
        total_final = sum(stats["final_size"] for stats in self.preprocessing_stats.values())
        overall_augmentation = (total_final - total_original) / total_original if total_original > 0 else 0

        summary_stats = {
            "total_datasets": len(self.preprocessing_stats),
            "total_original_rows": total_original,
            "total_final_rows": total_final,
            "overall_augmentation_ratio": overall_augmentation,
            "datasets": self.preprocessing_stats
        }

        # Log to wandb
        self.artifact.metadata.update({
            "preprocessing_summary": summary_stats
        })

        # Print summary
        print(f"Datasets processed: {len(self.preprocessing_stats)}")
        print(f"Total rows: {total_original:,} → {total_final:,}")
        print(f"Overall augmentation ratio: {overall_augmentation:.2%}")

        print("\nPer-dataset summary:")
        for dataset_name, stats in self.preprocessing_stats.items():
            print(f"  {dataset_name}: {stats['original_size']:,} → {stats['final_size']:,} "
                  f"({stats['augmentation_ratio']:.2%} augmentation)")

        print("="*60)

    def run_preprocessing(self) -> None:
        """Main method to initiate the preprocessing process."""
        print(f"Starting data preprocessing with config: {self.config_path}")
        print(f"Input directory: {self.input_folder_dir}")
        print(f"Output directory: {self.output_folder_dir}")

        self.process_all_datasets()
        print("\nData preprocessing completed!")

    def get_enabled_preprocessors(self) -> List[str]:
        """Get list of enabled preprocessor names."""
        return [
            preprocessor["name"]
            for preprocessor in self.config.get("preprocessors", [])
            if preprocessor.get("enabled", True)
        ]

    def get_enabled_augmenters(self) -> List[str]:
        """Get list of enabled augmenter names."""
        return [
            augmenter["name"]
            for augmenter in self.config.get("augmenters", [])
            if augmenter.get("enabled", True)
        ]

    def get_preprocessor_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured preprocessors."""
        info = {}
        for preprocessor in self.config.get("preprocessors", []):
            info[preprocessor["name"]] = {
                "description": preprocessor.get("description", ""),
                "enabled": preprocessor.get("enabled", True),
                "params": preprocessor.get("params", {})
            }
        return info

    def get_augmenter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured augmenters."""
        info = {}
        for augmenter in self.config.get("augmenters", []):
            info[augmenter["name"]] = {
                "description": augmenter.get("description", ""),
                "enabled": augmenter.get("enabled", True),
                "params": augmenter.get("params", {})
            }
        return info
