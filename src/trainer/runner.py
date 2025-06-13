"""Command line interface for launching training experiments."""

import os
from pathlib import Path
import click
import wandb

from src.trainer import NMTTrainer
from src.utils.utils import generate_folder_name, get_s3_loader


@click.command()
@click.option(
    "--splits-artifact-version",
    required=True,
    type=str,
    help="Version of the splits dataset artifact to use for training",
)
@click.option(
    "--bucket",
    default="personal-data-science-data",
    required=False,
    type=str,
    help="S3 bucket name where trained model will be saved",
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
    help="Description of this training run",
)
@click.option(
    "--config",
    default="config/training.yaml",
    required=False,
    type=str,
    help="Path to training configuration file",
)
@click.option(
    "--develop",
    is_flag=True,
    default=False,
    help="Run in development mode without logging to personal account",
)

def main(
    splits_artifact_version: str,
    bucket: str,
    project: str,
    description: str,
    config: str,
    develop: bool,
) -> None:
    """Run a training experiment based on configuration and dataset version.

    Parameters
    ----------
    splits_artifact_version:
        Version of the dataset splits artifact to download from Weights & Biases.
    bucket:
        Name of the S3 bucket to upload the final model to.
    project:
        WandB project name used for logging.
    description:
        Short description for this training run.
    config:
        Path to the YAML configuration file.
    develop:
        If ``True``, run in anonymous mode without using a personal account.
    """

    if develop:
        os.environ.pop("WANDB_API_KEY", None)

    for arg, value in locals().items():
        print(f"{arg}: {value}")

    with wandb.init(
        project=project,
        job_type="train-model",
        tags=["pipeline", "training"],
        notes=description,
        save_code=True,
        anonymous="must" if develop else "allow",
    ) as run:
        print(f"Downloading splits artifact: splits:{splits_artifact_version}")
        splits_artifact = run.use_artifact(f"splits:{splits_artifact_version}")
        data_dir = splits_artifact.download()
        print(f"Downloaded splits to: {data_dir}")

        artifact = wandb.Artifact(
            name="model",
            type="model",
            description=description,
        )

        output_folder_dir = Path("artifacts") / "models" / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        artifact.metadata.update({"output_dir": str(output_folder_dir)})

        s3_loader = get_s3_loader(bucket)

        trainer = NMTTrainer(
            artifact=artifact,
            dataset_dir=str(data_dir),
            config_path=config,
        )

        trainer.run_training()

        print(f"\nUploading model to S3: s3://{bucket}/{output_folder_dir}")
        s3_loader.upload(trainer.config.get("trainer", {}).get("output_dir", "model_output"))

        artifact.add_reference(f"s3://{bucket}/{str(output_folder_dir)}")
        run.log_artifact(artifact)

        print("\n✅ Training pipeline completed successfully!")
        print(f"   Source artifact: splits:{splits_artifact_version}")
        print(f"   Model saved to: {trainer.config.get('trainer', {}).get('output_dir', 'model_output')}")
        print(f"   S3 location: s3://{bucket}/{output_folder_dir}")


if __name__ == "__main__":
    main()
