import os
from pathlib import Path
import click
import wandb

from src.splitter.splitter import DataSplitter
from src.utils.utils import generate_folder_name, get_s3_loader


@click.command()
@click.option(
    "--bucket",
    default="personal-data-science-data",
    required=False,
    type=str,
    help="S3 bucket name where split data will be saved",
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
    help="Description of this splitting run",
)
@click.option(
    "--config",
    default="config/splitting.yaml",
    required=False,
    type=str,
    help="Path to splitting configuration file",
)
@click.option(
    "--develop",
    is_flag=True,
    default=False,
    help="Run in development mode without logging to personal account",
)
def main(
        bucket: str,
        project: str,
        description: str,
        config: str,
        develop: bool
) -> None:
    """
    Main function to run the data splitting pipeline.

    This pipeline splits data from various artifacts into train/valid/test sets
    based on YAML configuration, with optional contamination checking.

    Args:
        bucket: S3 bucket name where split data will be saved
        project: The name of the wandb project
        description: A description of the current splitting run
        config: Path to the splitting configuration file
        develop: If True, run in development mode without logging to personal account
    """

    # Remove API key if in develop mode
    if develop:
        os.environ.pop('WANDB_API_KEY', None)

    # Print all arguments for logging purposes
    print("=== Split Pipeline Configuration ===")
    for arg, value in locals().items():
        print(f"{arg}: {value}")
    print("=" * 35)

    # Initialize wandb run
    with wandb.init(
            project=project,
            job_type="split-data",
            tags=["pipeline", "splitting"],
            notes=description,
            save_code=True,
            anonymous="must" if develop else "allow"
    ) as run:

        # Create a new artifact
        artifact = wandb.Artifact(
            name="splits",
            type="dataset",
            description=description
        )

        # Create output directory
        output_folder_dir = Path('artifacts') / "splits" / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        artifact.metadata.update({'output_dir': str(output_folder_dir)})

        # Initialize S3 loader
        s3_loader = get_s3_loader(bucket)

        # Initialize splitter
        splitter = DataSplitter(
            artifact=artifact,
            output_folder_dir=str(output_folder_dir),
            config_path=config
        )

        # Print split info
        print("\nSplit Configuration:")
        for split_name, info in splitter.get_split_info().items():
            print(f"\n{split_name.upper()}:")
            for source in info['sources']:
                print(f"  - {source['percentage']}% from {source['artifact']}")

        # Print contamination check status
        contamination_config = splitter.config.get('contamination_check', {})
        if contamination_config.get('enabled', True):
            print("\nContamination checking: ENABLED")
            check_pairs = contamination_config.get('check_pairs', [])
            for pair in check_pairs:
                print(f"  - {pair['source']} → {pair['target']}: {pair.get('action', 'remove_from_target')}")
        else:
            print("\nContamination checking: DISABLED")
            print("  ⚠️  WARNING: No contamination checking will be performed!")

        # Run splitting
        splitter.run_splitting()

        # Upload to S3
        print(f"\nUploading results to S3: s3://{bucket}/{output_folder_dir}")
        s3_loader.upload(output_folder_dir)

        # Add S3 reference to artifact
        artifact.add_reference(f"s3://{bucket}/{str(output_folder_dir)}")

        # Log artifact to wandb
        run.log_artifact(artifact)

        # Get source artifacts used
        source_artifacts = set()
        for split_config in splitter.config.get('splits', {}).values():
            for source in split_config.get('sources', []):
                source_artifacts.add(source['artifact'])

        print(f"\n✅ Splitting pipeline completed successfully!")
        print(f"   Source artifacts used: {', '.join(sorted(source_artifacts))}")
        print(f"   Output saved to: {output_folder_dir}")
        print(f"   S3 location: s3://{bucket}/{output_folder_dir}")
        print(f"   Created splits: train.parquet, valid.parquet, test.parquet")

        # Log metadata about source artifacts
        artifact.metadata.update({
            'source_artifacts': list(source_artifacts),
            'num_source_artifacts': len(source_artifacts),
            'contamination_checking_enabled': contamination_config.get('enabled', True)
        })


if __name__ == '__main__':
    main()
