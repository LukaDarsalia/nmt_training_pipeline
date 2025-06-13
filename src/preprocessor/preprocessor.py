"""
Data Preprocessor with YAML configuration support.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time

import pandas as pd
import wandb
import yaml
from tqdm import tqdm

# Import all preprocessing and augmentation functions to register them
from .preprocessing_functions import *
from .augmentation_functions import *
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
        self.function_speeds = {}  # Track speeds for all functions

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
            "src/preprocessor/preprocessor.py",
            "src/preprocessor/preprocessing_functions.py",
            "src/preprocessor/augmentation_functions.py",
            "src/preprocessor/registry.py"
        ]

        for file_path in preprocessor_files:
            if Path(file_path).exists():
                self.artifact.add_file(file_path, name=f"preprocessor_code/{Path(file_path).name}")

    def _find_changed_rows(self,
                          original_df: pd.DataFrame,
                          processed_df: pd.DataFrame,
                          columns: List[str] = None) -> pd.DataFrame:
        """
        Find rows that were changed during preprocessing.

        Args:
            original_df: Original DataFrame
            processed_df: Processed DataFrame
            columns: Columns to check for changes (default: ['en', 'ka'])

        Returns:
            DataFrame containing only changed rows with before/after comparison
        """
        if columns is None:
            columns = ['en', 'ka']

        # Find rows where any of the specified columns changed
        changed_mask = pd.Series([False] * len(original_df))

        for col in columns:
            if col in original_df.columns and col in processed_df.columns:
                # Convert to string for comparison to handle NaN values
                original_col = original_df[col].astype(str)
                processed_col = processed_df[col].astype(str)
                changed_mask |= (original_col != processed_col)

        # Get changed rows
        changed_indices = original_df.index[changed_mask]

        if len(changed_indices) == 0:
            return pd.DataFrame()

        # Create comparison DataFrame
        comparison_data = []
        for idx in changed_indices:
            row_data = {'index': idx}

            # Add original and processed values for each column
            for col in columns:
                if col in original_df.columns:
                    row_data[f'{col}_before'] = original_df.loc[idx, col]
                    row_data[f'{col}_after'] = processed_df.loc[idx, col]

            # Add other columns for context
            for col in ['domain', 'id']:
                if col in original_df.columns:
                    row_data[col] = original_df.loc[idx, col]

            comparison_data.append(row_data)

        return pd.DataFrame(comparison_data)

    def _apply_single_preprocessor(self,
                                 df: pd.DataFrame,
                                 preprocessor_config: Dict[str, Any],
                                 dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply a single preprocessing function to the dataset.

        Args:
            df: Input DataFrame
            preprocessor_config: Configuration for the preprocessor
            dataset_name: Name of the dataset being processed

        Returns:
            Tuple of (processed DataFrame, stats dict)
        """
        if not preprocessor_config.get("enabled", True):
            print(f"  Skipping disabled preprocessor: {preprocessor_config['name']}")
            return df, {}

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

        # Keep a copy of original for comparison
        original_df = df.copy()

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
            rows_per_second = original_size / processing_time if processing_time > 0 else 0

            # Find changed rows
            changed_rows_df = self._find_changed_rows(
                original_df,
                processed_df,
                columns=preprocessor_config.get("params", {}).get("columns", ['en', 'ka'])
            )

            num_changed = len(changed_rows_df)
            change_percentage = (num_changed / original_size * 100) if original_size > 0 else 0

            # Prepare stats
            stats = {
                "processing_time_seconds": processing_time,
                "rows_per_second": rows_per_second,
                "total_rows": original_size,
                "rows_changed": num_changed,
                "change_percentage": change_percentage
            }

            # Store speed for later logging
            function_key = f"{dataset_name}_{preprocessor_name}"
            self.function_speeds[function_key] = stats

            # Log sample of changed rows
            if num_changed > 0:
                sample_size = min(10, num_changed)
                changed_sample = changed_rows_df.sample(sample_size, random_state=42) if num_changed > sample_size else changed_rows_df

                # Create wandb table
                table = wandb.Table(dataframe=changed_sample)
                self.artifact.add(table, f"{dataset_name}_{preprocessor_name}_changed_rows")

                # Log stats
                wandb.log({
                    f"{dataset_name}_{preprocessor_name}_speed": rows_per_second,
                    f"{dataset_name}_{preprocessor_name}_time": processing_time,
                    f"{dataset_name}_{preprocessor_name}_changed": num_changed,
                    f"{dataset_name}_{preprocessor_name}_change_pct": change_percentage
                })

            print(f"    ✓ {preprocessor_name}: Processed {original_size} rows in {processing_time:.1f}s "
                  f"({rows_per_second:.0f} rows/s, {num_changed} changed [{change_percentage:.1f}%])")

            return processed_df, stats

        except Exception as e:
            print(f"    ✗ Error in {preprocessor_name}: {str(e)}")
            raise

    def _apply_single_augmenter(self,
                              df: pd.DataFrame,
                              augmenter_config: Dict[str, Any],
                              dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply a single augmentation function to the dataset.

        Args:
            df: Input DataFrame
            augmenter_config: Configuration for the augmenter
            dataset_name: Name of the dataset being augmented

        Returns:
            Tuple of (DataFrame with augmented examples, stats dict)
        """
        if not augmenter_config.get("enabled", True):
            print(f"  Skipping disabled augmenter: {augmenter_config['name']}")
            return pd.DataFrame(), {}

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
        input_size = len(df)

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
            rows_per_second = input_size / processing_time if processing_time > 0 else 0
            examples_per_second = augmented_size / processing_time if processing_time > 0 else 0

            # Prepare stats
            stats = {
                "processing_time_seconds": processing_time,
                "input_rows_per_second": rows_per_second,
                "generated_examples_per_second": examples_per_second,
                "input_rows": input_size,
                "generated_examples": augmented_size,
                "generation_ratio": augmented_size / input_size if input_size > 0 else 0
            }

            # Store speed for later logging
            function_key = f"{dataset_name}_{augmenter_name}"
            self.function_speeds[function_key] = stats

            # Log sample of augmented data
            if augmented_size > 0:
                sample_size = min(10, augmented_size)
                augmented_sample = augmented_df.sample(sample_size, random_state=42)

                # Create wandb table
                table = wandb.Table(dataframe=augmented_sample)
                self.artifact.add(table, f"{dataset_name}_{augmenter_name}_sample")

                # Log stats
                wandb.log({
                    f"{dataset_name}_{augmenter_name}_speed": rows_per_second,
                    f"{dataset_name}_{augmenter_name}_time": processing_time,
                    f"{dataset_name}_{augmenter_name}_generated": augmented_size,
                    f"{dataset_name}_{augmenter_name}_gen_per_sec": examples_per_second
                })

            print(f"    ✓ {augmenter_name}: Generated {augmented_size} examples in {processing_time:.1f}s "
                  f"({rows_per_second:.0f} input rows/s, {examples_per_second:.0f} examples/s)")

            return augmented_df, stats

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
            "augmentation_ratio": 0.0,
            "total_preprocessing_time": 0.0,
            "total_augmentation_time": 0.0,
            "final_cleanup_stats": {}
        }

        current_df = df.copy()

        # Apply preprocessing steps
        print("  --- Preprocessing Steps ---")
        for preprocessor_config in self.config.get("preprocessors", []):
            if preprocessor_config.get("enabled", True):
                processed_df, stats = self._apply_single_preprocessor(
                    current_df, preprocessor_config, dataset_name
                )
                current_df = processed_df

                # Store preprocessor stats
                self.preprocessing_stats[dataset_name]["preprocessors"][preprocessor_config["name"]] = stats
                self.preprocessing_stats[dataset_name]["total_preprocessing_time"] += stats.get(
                    "processing_time_seconds", 0)

        # Apply augmentation steps
        print("  --- Augmentation Steps ---")
        all_augmented_data = []
        dataset_restructured = False

        for augmenter_config in self.config.get("augmenters", []):
            if augmenter_config.get("enabled", True):
                augmenter_name = augmenter_config["name"]

                # Special handling for dataset restructuring operations
                if augmenter_name == "restructure_to_longer_texts":
                    # This replaces the dataset rather than adding to it
                    restructured_df, stats = self._apply_single_augmenter(
                        current_df, augmenter_config, dataset_name
                    )
                    if len(restructured_df) > 0:
                        current_df = restructured_df  # Replace the dataset
                        dataset_restructured = True
                        self.preprocessing_stats[dataset_name]["augmenters"][augmenter_name] = stats
                        self.preprocessing_stats[dataset_name]["total_augmentation_time"] += stats.get(
                            "processing_time_seconds", 0)
                        print(f"    ✓ Dataset restructured by {augmenter_name}")
                else:
                    # Normal augmentation - adds new examples
                    augmented_df, stats = self._apply_single_augmenter(
                        current_df, augmenter_config, dataset_name
                    )

                    if len(augmented_df) > 0:
                        all_augmented_data.append(augmented_df)
                        self.preprocessing_stats[dataset_name]["augmenters"][augmenter_name] = stats
                        self.preprocessing_stats[dataset_name]["total_augmentation_time"] += stats.get(
                            "processing_time_seconds", 0)

        # Combine original and augmented data (if not restructured)
        if all_augmented_data and not dataset_restructured:
            combined_augmented = pd.concat(all_augmented_data, ignore_index=True)
            combined_df = pd.concat([current_df, combined_augmented], ignore_index=True)
            print(f"  Combined {len(combined_augmented)} augmented examples with {len(current_df)} original")
        elif all_augmented_data and dataset_restructured:
            # Dataset was restructured, but we still have normal augmentation
            combined_augmented = pd.concat(all_augmented_data, ignore_index=True)
            combined_df = pd.concat([current_df, combined_augmented], ignore_index=True)
            print(f"  Combined {len(combined_augmented)} augmented examples with {len(current_df)} restructured")
        else:
            combined_df = current_df
            if dataset_restructured:
                print("  Dataset restructured, no additional augmented examples")
            else:
                print("  No augmented examples generated")

        # Perform final cleanup: remove empty content and duplicates
        final_df, cleanup_stats = self._final_cleanup(combined_df, dataset_name)
        self.preprocessing_stats[dataset_name]["final_cleanup_stats"] = cleanup_stats

        # Update final statistics
        final_size = len(final_df)
        augmentation_ratio = (len(combined_df) - len(current_df)) / len(current_df) if len(current_df) > 0 else 0

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
            f"{dataset_name}_combined_size": len(combined_df),
            f"{dataset_name}_final_size": final_size,
            f"{dataset_name}_empty_content_removed": cleanup_stats["empty_content_removed"],
            f"{dataset_name}_duplicates_removed": cleanup_stats["duplicates_removed"],
            f"{dataset_name}_total_cleanup_removed": cleanup_stats["total_removed"],
            f"{dataset_name}_cleanup_retention_rate": cleanup_stats["retention_rate"],
            f"{dataset_name}_augmentation_ratio": augmentation_ratio,
            f"{dataset_name}_total_preprocessing_time": self.preprocessing_stats[dataset_name][
                "total_preprocessing_time"],
            f"{dataset_name}_total_augmentation_time": self.preprocessing_stats[dataset_name][
                "total_augmentation_time"],
            f"{dataset_name}_cleanup_time": cleanup_stats["processing_time_seconds"]
        })

        print(f"  ✓ Saved processed dataset: {output_path}")
        print(f"  📊 Final: {len(df)} → {len(combined_df)} → {final_size} "
              f"({augmentation_ratio:.2%} augmentation, {cleanup_stats['total_removed']} rows cleaned up)")

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
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)

        # Create summary statistics
        total_original = sum(stats["original_size"] for stats in self.preprocessing_stats.values())
        total_final = sum(stats["final_size"] for stats in self.preprocessing_stats.values())
        total_empty_removed = sum(
            stats["final_cleanup_stats"]["empty_content_removed"] for stats in self.preprocessing_stats.values())
        total_duplicates_removed = sum(
            stats["final_cleanup_stats"]["duplicates_removed"] for stats in self.preprocessing_stats.values())
        total_cleanup_removed = sum(
            stats["final_cleanup_stats"]["total_removed"] for stats in self.preprocessing_stats.values())
        overall_augmentation = (
                                       total_final + total_cleanup_removed - total_original) / total_original if total_original > 0 else 0

        total_preprocessing_time = sum(stats["total_preprocessing_time"] for stats in self.preprocessing_stats.values())
        total_augmentation_time = sum(stats["total_augmentation_time"] for stats in self.preprocessing_stats.values())
        total_cleanup_time = sum(
            stats["final_cleanup_stats"]["processing_time_seconds"] for stats in self.preprocessing_stats.values())
        total_time = total_preprocessing_time + total_augmentation_time + total_cleanup_time

        summary_stats = {
            "total_datasets": len(self.preprocessing_stats),
            "total_original_rows": total_original,
            "total_final_rows": total_final,
            "total_empty_content_removed": total_empty_removed,
            "total_duplicates_removed": total_duplicates_removed,
            "total_cleanup_removed": total_cleanup_removed,
            "overall_cleanup_percentage": (total_cleanup_removed / (total_final + total_cleanup_removed) * 100) if (
                                                                                                                           total_final + total_cleanup_removed) > 0 else 0,
            "overall_augmentation_ratio": overall_augmentation,
            "total_preprocessing_time_seconds": total_preprocessing_time,
            "total_augmentation_time_seconds": total_augmentation_time,
            "total_cleanup_time_seconds": total_cleanup_time,
            "total_processing_time_seconds": total_time,
            "overall_rows_per_second": total_original / total_time if total_time > 0 else 0,
            "datasets": self.preprocessing_stats
        }

        # Log to wandb
        self.artifact.metadata.update({
            "preprocessing_summary": summary_stats
        })

        # Create speed summary table
        speed_data = []
        for function_key, stats in self.function_speeds.items():
            speed_data.append({
                "function": function_key,
                "time_seconds": stats.get("processing_time_seconds", 0),
                "rows_per_second": stats.get("rows_per_second", 0),
                "rows_changed": stats.get("rows_changed", 0),
                "examples_generated": stats.get("generated_examples", 0)
            })

        speed_df = pd.DataFrame(speed_data)
        if len(speed_df) > 0:
            speed_table = wandb.Table(dataframe=speed_df)
            wandb.log({"function_speed_summary": speed_table})

        # Print summary
        print(f"Datasets processed: {len(self.preprocessing_stats)}")
        print(f"Total rows: {total_original:,} → {total_final:,}")
        print(f"Cleanup removed: {total_cleanup_removed:,} rows")
        print(f"  - Empty content: {total_empty_removed:,}")
        print(f"  - Duplicates: {total_duplicates_removed:,}")
        print(f"Overall augmentation ratio: {overall_augmentation:.2%}")
        print(f"Total processing time: {total_time:.1f}s")
        print(f"  - Preprocessing: {total_preprocessing_time:.1f}s")
        print(f"  - Augmentation: {total_augmentation_time:.1f}s")
        print(f"  - Final cleanup: {total_cleanup_time:.1f}s")
        print(f"Overall speed: {total_original / total_time if total_time > 0 else 0:.0f} rows/s")

        print("\nPer-dataset summary:")
        for dataset_name, stats in self.preprocessing_stats.items():
            cleanup_stats = stats["final_cleanup_stats"]
            total_dataset_time = (stats['total_preprocessing_time'] +
                                  stats['total_augmentation_time'] +
                                  cleanup_stats['processing_time_seconds'])

            print(f"  {dataset_name}: {stats['original_size']:,} → {stats['final_size']:,} "
                  f"({stats['augmentation_ratio']:.2%} augmentation, "
                  f"{cleanup_stats['total_removed']} cleaned up "
                  f"[{cleanup_stats['empty_content_removed']} empty + {cleanup_stats['duplicates_removed']} duplicates], "
                  f"{total_dataset_time:.1f}s)")

        print("=" * 60)

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

    def _remove_duplicates(self,
                           df: pd.DataFrame,
                           dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove duplicate rows where both 'en' and 'ka' columns are identical.

        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset for logging

        Returns:
            Tuple of (deduplicated DataFrame, stats dict)
        """
        print(f"  --- Duplicate Removal ---")
        print(f"  Removing duplicates based on 'en' and 'ka' columns...")

        start_time = time.time()
        original_size = len(df)

        # Find duplicates based on both 'en' and 'ka' columns
        duplicate_mask = df.duplicated(subset=['en', 'ka'], keep='first')
        duplicate_rows = df[duplicate_mask].copy()

        # Remove duplicates (keep first occurrence)
        deduplicated_df = df.drop_duplicates(subset=['en', 'ka'], keep='first').reset_index(drop=True)

        # Calculate statistics
        processing_time = time.time() - start_time
        final_size = len(deduplicated_df)
        num_duplicates = original_size - final_size
        duplicate_percentage = (num_duplicates / original_size * 100) if original_size > 0 else 0

        stats = {
            "processing_time_seconds": processing_time,
            "original_rows": original_size,
            "final_rows": final_size,
            "duplicates_removed": num_duplicates,
            "duplicate_percentage": duplicate_percentage,
            "retention_rate": final_size / original_size if original_size > 0 else 1.0
        }

        # Log duplicate samples if any were found
        if num_duplicates > 0:
            sample_size = min(20, len(duplicate_rows))
            duplicate_sample = duplicate_rows.sample(sample_size, random_state=42) if len(
                duplicate_rows) > sample_size else duplicate_rows

            # Add some context columns for better understanding
            log_columns = ['en', 'ka', 'domain', 'id']
            available_columns = [col for col in log_columns if col in duplicate_sample.columns]

            # Create wandb table with duplicate examples
            table = wandb.Table(dataframe=duplicate_sample[available_columns])
            self.artifact.add(table, f"{dataset_name}_duplicate_examples")

            # Log duplicate statistics to wandb
            wandb.log({
                f"{dataset_name}_duplicates_removed": num_duplicates,
                f"{dataset_name}_duplicate_percentage": duplicate_percentage,
                f"{dataset_name}_dedup_retention_rate": stats["retention_rate"]
            })

            print(f"    ✓ Found and removed {num_duplicates} duplicate rows ({duplicate_percentage:.2f}%)")
            print(f"    ✓ Retention rate: {stats['retention_rate']:.2%}")
        else:
            print(f"    ✓ No duplicates found")

        print(f"    ✓ Deduplication completed in {processing_time:.1f}s")

        return deduplicated_df, stats

    def _final_cleanup(self,
                       df: pd.DataFrame,
                       dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform final cleanup: remove empty content and duplicate rows.

        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset for logging

        Returns:
            Tuple of (cleaned DataFrame, stats dict)
        """
        print(f"  --- Final Cleanup ---")

        start_time = time.time()
        original_size = len(df)
        current_df = df.copy()

        # Step 1: Remove rows with empty content in 'en' or 'ka' columns
        print(f"  Removing rows with empty 'en' or 'ka' content...")

        def is_empty_content(text: str) -> bool:
            """Check if text is empty, whitespace-only, or newlines-only."""
            if pd.isna(text) or text is None:
                return True

            # Convert to string and check if it's empty after stripping whitespace and newlines
            cleaned_text = str(text).strip().replace('\n', '').replace('\r', '').replace('\t', '')
            return len(cleaned_text) == 0

        # Create mask for rows with empty content
        empty_en_mask = current_df['en'].apply(is_empty_content)
        empty_ka_mask = current_df['ka'].apply(is_empty_content)
        empty_content_mask = empty_en_mask | empty_ka_mask

        # Get empty rows for logging
        empty_rows = current_df[empty_content_mask].copy()
        num_empty = len(empty_rows)

        # Remove empty content rows
        current_df = current_df[~empty_content_mask].reset_index(drop=True)
        after_empty_removal = len(current_df)

        # Step 2: Remove duplicates based on both 'en' and 'ka' columns
        print(f"  Removing duplicates based on 'en' and 'ka' columns...")

        # Find duplicates
        duplicate_mask = current_df.duplicated(subset=['en', 'ka'], keep='first')
        duplicate_rows = current_df[duplicate_mask].copy()
        num_duplicates = len(duplicate_rows)

        # Remove duplicates (keep first occurrence)
        final_df = current_df.drop_duplicates(subset=['en', 'ka'], keep='first').reset_index(drop=True)

        # Calculate final statistics
        processing_time = time.time() - start_time
        final_size = len(final_df)

        total_removed = original_size - final_size
        empty_percentage = (num_empty / original_size * 100) if original_size > 0 else 0
        duplicate_percentage = (num_duplicates / after_empty_removal * 100) if after_empty_removal > 0 else 0
        total_removal_percentage = (total_removed / original_size * 100) if original_size > 0 else 0

        stats = {
            "processing_time_seconds": processing_time,
            "original_rows": original_size,
            "final_rows": final_size,
            "empty_content_removed": num_empty,
            "empty_content_percentage": empty_percentage,
            "duplicates_removed": num_duplicates,
            "duplicate_percentage": duplicate_percentage,
            "total_removed": total_removed,
            "total_removal_percentage": total_removal_percentage,
            "retention_rate": final_size / original_size if original_size > 0 else 1.0
        }

        # Log empty content samples if any were found
        if num_empty > 0:
            sample_size = min(10, len(empty_rows))
            empty_sample = empty_rows.sample(sample_size, random_state=42) if len(
                empty_rows) > sample_size else empty_rows

            # Add some context columns for better understanding
            log_columns = ['en', 'ka', 'domain', 'id']
            available_columns = [col for col in log_columns if col in empty_sample.columns]

            # Create wandb table with empty content examples
            table = wandb.Table(dataframe=empty_sample[available_columns])
            self.artifact.add(table, f"{dataset_name}_empty_content_examples")

            print(f"    ✓ Found and removed {num_empty} rows with empty content ({empty_percentage:.2f}%)")
        else:
            print(f"    ✓ No empty content found")

        # Log duplicate samples if any were found
        if num_duplicates > 0:
            sample_size = min(10, len(duplicate_rows))
            duplicate_sample = duplicate_rows.sample(sample_size, random_state=42) if len(
                duplicate_rows) > sample_size else duplicate_rows

            # Add some context columns for better understanding
            log_columns = ['en', 'ka', 'domain', 'id']
            available_columns = [col for col in log_columns if col in duplicate_sample.columns]

            # Create wandb table with duplicate examples
            table = wandb.Table(dataframe=duplicate_sample[available_columns])
            self.artifact.add(table, f"{dataset_name}_duplicate_examples")

            print(f"    ✓ Found and removed {num_duplicates} duplicate rows ({duplicate_percentage:.2f}%)")
        else:
            print(f"    ✓ No duplicates found")

        # Log cleanup statistics to wandb
        wandb.log({
            f"{dataset_name}_empty_content_removed": num_empty,
            f"{dataset_name}_empty_content_percentage": empty_percentage,
            f"{dataset_name}_duplicates_removed": num_duplicates,
            f"{dataset_name}_duplicate_percentage": duplicate_percentage,
            f"{dataset_name}_total_cleanup_removed": total_removed,
            f"{dataset_name}_cleanup_retention_rate": stats["retention_rate"]
        })

        print(f"    ✓ Total cleanup: {total_removed} rows removed ({total_removal_percentage:.2f}%)")
        print(f"    ✓ Final retention rate: {stats['retention_rate']:.2%}")
        print(f"    ✓ Cleanup completed in {processing_time:.1f}s")

        return final_df, stats
