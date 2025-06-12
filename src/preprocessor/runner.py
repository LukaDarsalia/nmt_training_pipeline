import os
from pathlib import Path
import click
import wandb

from src.preprocessor import DataPreprocessor
from src.utils.utils import generate_folder_name, get_s3_loader


@click.command()
@click.option(
    "--cleaned-artifact-version",
    required=True,
    type=str,
    help="Version of the cleaned dataset artifact to preprocess (e.g., 'latest', 'v1', etc.)",
)
@click.option(
    "--bucket",
    default="personal-data-science-data",
    required=False,
    type=str,
    help="S3 bucket name where preprocessed data will be saved",
)
@click.option(
    "--project",
    default="NMT_Training",
    required=False,
    type=str,
    help="WandB project name to log to",
)
@click.option(
    "--description",
    required=True,
    type=str,
    help="Description of this preprocessing run",
)
@click.option(
    "--config",
    default="config/preprocessing.yaml",
    required=False,
    type=str,
    help="Path to preprocessing configuration file",
)
@click.option(
    "--develop",
    is_flag=True,
    default=False,
    help="Run in development mode without logging to personal account",
)
def main(
        cleaned_artifact_version: str,
        bucket: str,
        project: str,
        description: str,
        config: str,
        develop: bool
) -> None:
    """
    Main function to run the data preprocessing pipeline.

    Args:
        cleaned_artifact_version: Version of the cleaned dataset artifact to preprocess
        bucket: S3 bucket name where preprocessed data will be saved
        project: The name of the wandb project
        description: A description of the current preprocessing run
        config: Path to the preprocessing configuration file
        develop: If True, run in development mode without logging to personal account
    """

    # Remove API key if in develop mode
    if develop:
        os.environ.pop('WANDB_API_KEY', None)

    # Print all arguments for logging purposes
    for arg, value in locals().items():
        print(f"{arg}: {value}")

    # Initialize wandb run
    with wandb.init(
            project=project,
            job_type="preprocess-data",
            tags=["pipeline", "preprocessing", "augmentation"],
            notes=description,
            save_code=True,
            anonymous="must" if develop else "allow"
    ) as run:

        # ------------------ Download cleaned dataset artifact ------------------
        print(f"Downloading cleaned dataset artifact: cleaned:{cleaned_artifact_version}")
        cleaned_dataset_artifact = run.use_artifact(f"cleaned:{cleaned_artifact_version}")
        input_data_dir = cleaned_dataset_artifact.download()
        print(f"Downloaded cleaned data to: {input_data_dir}")

        # Validate downloaded data contains parquet files
        input_path = Path(input_data_dir)
        parquet_files = list(input_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in downloaded artifact: {input_data_dir}")

        print(f"Found {len(parquet_files)} parquet files to preprocess:")
        for file in parquet_files:
            print(f"  - {file.name}")

        # Create a new artifact
        artifact = wandb.Artifact(
            name="preprocessed",
            type="dataset",
            description=description
        )

        # Create output directory
        output_folder_dir = Path('artifacts') / "preprocessed" / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        artifact.metadata.update({'output_dir': str(output_folder_dir)})

        # Initialize S3 loader
        s3_loader = get_s3_loader(bucket)

        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            artifact=artifact,
            input_folder_dir=str(input_path),
            output_folder_dir=str(output_folder_dir),
            config_path=config
        )

        # Print preprocessor info
        print("\nPreprocessing Configuration:")
        preprocessor_info = preprocessor.get_preprocessor_info()
        for name, info in preprocessor_info.items():
            status = "✓ Enabled" if info["enabled"] else "✗ Disabled"
            params = f" (params: {info['params']})" if info['params'] else ""
            print(f"  {name}: {status} - {info['description']}{params}")

        print("\nAugmentation Configuration:")
        augmenter_info = preprocessor.get_augmenter_info()
        for name, info in augmenter_info.items():
            status = "✓ Enabled" if info["enabled"] else "✗ Disabled"
            params = f" (params: {info['params']})" if info['params'] else ""
            print(f"  {name}: {status} - {info['description']}{params}")

        # Run preprocessing
        preprocessor.run_preprocessing()

        # Upload to S3
        print(f"\nUploading results to S3: s3://{bucket}/{output_folder_dir}")
        s3_loader.upload(output_folder_dir)

        # Add S3 reference to artifact
        artifact.add_reference(f"s3://{bucket}/{str(output_folder_dir)}")

        # Log artifact to wandb
        run.log_artifact(artifact)

        print(f"\n✅ Preprocessing pipeline completed successfully!")
        print(f"   Source artifact: cleaned:{cleaned_artifact_version}")
        print(f"   Output saved to: {output_folder_dir}")
        print(f"   S3 location: s3://{bucket}/{output_folder_dir}")
        print(f"   Processed {len(parquet_files)} dataset(s)")

        # Log metadata about source artifact
        artifact.metadata.update({
            'source_artifact': f"cleaned:{cleaned_artifact_version}",
            'input_datasets_count': len(parquet_files),
            'input_dataset_names': [f.stem for f in parquet_files]
        })


if __name__ == '__main__':
    main()
