"""
Updated DataLoader with YAML configuration support.
"""

from pathlib import Path
from typing import List

import wandb
import yaml
from tqdm import tqdm

from .loaders import *  # Import all loader functions to register them
from .registry import loader_registry

tqdm.pandas()


class DataLoader:
    """
    A class for loading and processing various datasets based on YAML configuration,
    generating appropriate dataset objects, and logging metadata and samples to wandb.
    """

    def __init__(self, artifact: wandb.Artifact, output_folder_dir: str, config_path: str = "config/datasets.yaml"):
        """
        Initialize the DataLoader.

        Args:
            artifact (wandb.Artifact): The wandb artifact to log data to.
            output_folder_dir (str): The directory to save processed datasets.
            config_path (str): Path to the YAML configuration file.
        """
        self.artifact = artifact
        self.output_folder_dir = Path(output_folder_dir)
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Log the config file to wandb
        self._log_config_to_wandb()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _log_config_to_wandb(self) -> None:
        """Log configuration and loader files to wandb."""
        # Log config file
        self.artifact.add_file(str(self.config_path), name="config.yaml")

        # Log loader files
        loader_files = [
            "src/loader/loader.py",
            "src/loader/loaders.py",
            "src/loader/registry.py"
        ]

        for file_path in loader_files:
            if Path(file_path).exists():
                self.artifact.add_file(file_path, name=f"loader_code/{Path(file_path).name}")

    def _load_single_dataset(self, dataset_config: Dict[str, Any]) -> None:
        """
        Load a single dataset based on its configuration.

        Args:
            dataset_config: Configuration for a single dataset
        """
        if not dataset_config.get("enabled", True):
            print(f"Skipping disabled dataset: {dataset_config['name']}")
            return

        print(f"Loading dataset: {dataset_config['name']}")

        # Get the loader function
        loader_function_name = dataset_config["loader_function"]
        loader_function = loader_registry.get_loader(loader_function_name)

        if loader_function is None:
            raise ValueError(f"Loader function '{loader_function_name}' not found. "
                           f"Available loaders: {list(loader_registry.list_loaders().keys())}")

        # Load the data
        try:
            df = loader_function(dataset_config)

            # Validate output format
            df = loader_registry.validate_loader_output(df, dataset_config['name'])

            # Save to parquet
            output_path = self.output_folder_dir / f"{dataset_config['name']}.parquet"
            df.to_parquet(output_path, index=False)

            # Log sample to wandb
            sample_size = min(10, len(df))
            sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
            table = wandb.Table(dataframe=sample)
            self.artifact.add(table, f"{dataset_config['name']}_sample")

            # Update metadata
            self.artifact.metadata.update({
                f"{dataset_config['name']}_size": len(df),
                f"{dataset_config['name']}_description": dataset_config.get("description", ""),
                f"{dataset_config['name']}_source": dataset_config.get("source", "")
            })

            print(f"Successfully loaded {len(df)} rows for {dataset_config['name']}")

        except Exception as e:
            print(f"Error loading dataset {dataset_config['name']}: {str(e)}")
            raise

    def load_all_data(self) -> None:
        """Load all enabled datasets from configuration."""
        datasets = self.config.get("datasets", [])

        if not datasets:
            print("No datasets found in configuration")
            return

        print(f"Found {len(datasets)} datasets in configuration")
        print(f"Available loaders: {list(loader_registry.list_loaders().keys())}")

        for dataset_config in datasets:
            self._load_single_dataset(dataset_config)

    def run_loading(self) -> None:
        """Main method to initiate the data loading process."""
        print(f"Starting data loading with config: {self.config_path}")
        self.load_all_data()
        print("Data loading completed!")

    def get_enabled_datasets(self) -> List[str]:
        """Get list of enabled dataset names."""
        return [ds["name"] for ds in self.config.get("datasets", []) if ds.get("enabled", True)]

    def get_dataset_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured datasets."""
        info = {}
        for ds in self.config.get("datasets", []):
            info[ds["name"]] = {
                "description": ds.get("description", ""),
                "source": ds.get("source", ""),
                "enabled": ds.get("enabled", True),
                "loader_function": ds.get("loader_function", "")
            }
        return info
