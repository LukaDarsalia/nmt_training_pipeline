import os
from pathlib import Path
import click
import wandb

from src.cleaner.cleaner import DataCleaner
from src.utils.utils import generate_folder_name, get_s3_loader


@click.command()
@click.option(
    "--raw-artifact-version",
    required=True,
    type=str,
    help="Version of the raw dataset artifact to clean (e.g., 'latest', 'v1', etc.)",
)
@click.option(
    "--bucket",
    default="personal-data-science-data",
    required=False,
    type=str,
    help="S3 bucket name where cleaned data will be saved",
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
    help="Description of this cleaning run",
)
@click.option(
    "--config",
    default="config/cleaning.yaml",
    required=False,
    type=str,
    help="Path to cleaning configuration file",
)
@click.option(
    "--develop",
    is_flag=True,
    default=False,
    help="Run in development mode without logging to personal account",
)
def main(
        raw_artifact_version: str,
        bucket: str,
        project: str,
        description: str,
        config: str,
        develop: bool
) -> None:
    """
    Main function to run the data cleaning pipeline.

    Args:
        raw_artifact_version: Version of the raw dataset artifact to clean
        bucket: S3 bucket name where cleaned data will be saved
        project: The name of the wandb project
        description: A description of the current cleaning run
        config: Path to the cleaning configuration file
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
            job_type="clean-data",
            tags=["pipeline", "cleaning"],
            notes=description,
            save_code=True,
            anonymous="must" if develop else "allow"
    ) as run:

        # ------------------ Download raw dataset artifact ------------------
        print(f"Downloading raw dataset artifact: raw:{raw_artifact_version}")
        raw_dataset_artifact = run.use_artifact(f"raw:{raw_artifact_version}")
        input_data_dir = raw_dataset_artifact.download()
        print(f"Downloaded raw data to: {input_data_dir}")

        # Validate downloaded data contains parquet files
        input_path = Path(input_data_dir)
        parquet_files = list(input_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in downloaded artifact: {input_data_dir}")

        print(f"Found {len(parquet_files)} parquet files to clean:")
        for file in parquet_files:
            print(f"  - {file.name}")

        # Create a new artifact
        artifact = wandb.Artifact(
            name="cleaned",
            type="dataset",
            description=description
        )

        # Create output directory
        output_folder_dir = Path('artifacts') / "cleaned" / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        artifact.metadata.update({'output_dir': str(output_folder_dir)})

        # Initialize S3 loader
        s3_loader = get_s3_loader(bucket)

        # Initialize cleaner
        cleaner = DataCleaner(
            artifact=artifact,
            input_folder_dir=str(input_path),
            output_folder_dir=str(output_folder_dir),
            config_path=config
        )

        # Print cleaner info
        print("\nCleaning Configuration:")
        for name, info in cleaner.get_cleaner_info().items():
            status = "✓ Enabled" if info["enabled"] else "✗ Disabled"
            params = f" (params: {info['params']})" if info['params'] else ""
            print(f"  {name}: {status} - {info['description']}{params}")

        # Run cleaning
        cleaner.run_cleaning()

        # Upload to S3
        print(f"\nUploading results to S3: s3://{bucket}/{output_folder_dir}")
        s3_loader.upload(output_folder_dir)

        # Add S3 reference to artifact
        artifact.add_reference(f"s3://{bucket}/{str(output_folder_dir)}")

        # Log artifact to wandb
        run.log_artifact(artifact)

        print(f"\n✅ Cleaning pipeline completed successfully!")
        print(f"   Source artifact: raw:{raw_artifact_version}")
        print(f"   Output saved to: {output_folder_dir}")
        print(f"   S3 location: s3://{bucket}/{output_folder_dir}")
        print(f"   Processed {len(parquet_files)} dataset(s)")

        # Log metadata about source artifact
        artifact.metadata.update({
            'source_artifact': f"raw:{raw_artifact_version}",
            'input_datasets_count': len(parquet_files),
            'input_dataset_names': [f.stem for f in parquet_files]
        })


if __name__ == '__main__':
    main()
