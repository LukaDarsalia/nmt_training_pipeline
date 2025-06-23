"""
Training Pipeline Runner

CLI script for running neural machine translation training experiments.
Uses YAML configuration files and registry system for flexible experimentation.
"""

import os
from pathlib import Path
from typing import Dict, Any

import click
import wandb
import yaml

from .trainer import NMTTrainer
from ..utils.utils import generate_folder_name, get_s3_loader

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML configuration: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['model', 'data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section '{section}' not found")

    # Check model configuration
    model_config = config['model']
    if 'type' not in model_config:
        raise ValueError("Model type must be specified in config.model.type")

    # Validate model type exists in registry
    from .registry import model_registry
    if model_config['type'] not in model_registry:
        available_models = list(model_registry.list_components().keys())
        raise ValueError(
            f"Model type '{model_config['type']}' not found. "
            f"Available models: {available_models}"
        )

    # Check trainer configuration if specified
    trainer_config = config.get('trainer', {})
    if 'type' in trainer_config:
        from .registry import trainer_registry
        if trainer_config['type'] not in trainer_registry:
            available_trainers = list(trainer_registry.list_components().keys())
            raise ValueError(
                f"Trainer type '{trainer_config['type']}' not found. "
                f"Available trainers: {available_trainers}"
            )

    print("Configuration validation passed")


@click.command()
@click.option(
    "--config",
    required=True,
    type=str,
    help="Path to training configuration YAML file"
)
@click.option(
    "--splits-artifact-version",
    required=True,
    type=str,
    help="Version of the splits artifact to use"
)
@click.option(
    "--model-artifact-version",
    required=False,
    type=str,
    help="Version of the model artifact to use for fine-tuning (optional)"
)
@click.option(
    "--bucket",
    default="personal-data-science-data",
    required=False,
    type=str,
    help="S3 bucket name where outputs will be saved"
)
@click.option(
    "--project",
    default="NMT_Training",
    required=False,
    type=str,
    help="WandB project name for logging"
)
@click.option(
    "--run-name",
    required=False,
    type=str,
    help="Optional name for the WandB run"
)
@click.option(
    "--description",
    required=True,
    type=str,
    help="Description of this training experiment"
)
@click.option(
    "--develop",
    is_flag=True,
    default=False,
    help="Run in development mode without logging to personal account"
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate configuration and setup without running training"
)
def main(
        config: str,
        splits_artifact_version: str,
        model_artifact_version: str,
        bucket: str,
        project: str,
        run_name: str,
        description: str,
        develop: bool,
        dry_run: bool
) -> None:
    """
    Run neural machine translation training experiment.

    This script orchestrates the complete training pipeline using the registry
    system for flexible experimentation with different models, trainers, and
    evaluation metrics.

    Example:
        python -m src.training.runner \\
            --config config/training/experiment_1.yaml \\
            --splits-artifact-version "latest" \\
            --description "Baseline Marian model experiment"
    """

    # Remove API key if in develop mode
    if develop:
        os.environ.pop('WANDB_API_KEY', None)
        print("Running in development mode (offline)")

    # Print configuration
    print("=" * 70)
    print("NEURAL MACHINE TRANSLATION TRAINING PIPELINE")
    print("=" * 70)
    print(f"Configuration file: {config}")
    print(f"Splits artifact: splits:{splits_artifact_version}")
    if model_artifact_version:
        print(f"Model artifact: training:{model_artifact_version}")
    print(f"Project: {project}")
    print(f"Description: {description}")
    print(f"Develop mode: {develop}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)

    try:
        # Load and validate configuration
        print("\n📋 Loading configuration...")
        training_config = load_config(config)
        validate_config(training_config)

        # Print available components
        print("\n🔧 Available components:")
        from .registry import model_registry, trainer_registry, evaluator_registry

        print(f"Models: {list(model_registry.list_components().keys())}")
        print(f"Trainers: {list(trainer_registry.list_components().keys())}")
        print(f"Evaluators: {list(evaluator_registry.list_components().keys())}")

        # Initialize WandB
        print("\n📊 Initializing WandB...")
        with wandb.init(
                project=project,
                job_type="training",
                tags=["pipeline", "nmt", training_config['model']['type']],
                notes=description,
                name=run_name,
                config=training_config,
                save_code=True,
                anonymous="must" if develop else "allow"
        ) as run:

            # Create artifact
            artifact = wandb.Artifact(
                name="training",
                type="model",
                description=description,
                metadata=training_config
            )

            # Download artifacts
            print("\n📥 Downloading artifacts...")

            # Download splits
            splits_artifact = run.use_artifact(f"splits:{splits_artifact_version}")
            splits_dir = splits_artifact.download()
            print(f"Downloaded splits to: {splits_dir}")

            # Download model if specified
            model_dir = None
            if model_artifact_version:
                model_artifact = run.use_artifact(f"training:{model_artifact_version}")
                model_dir = model_artifact.download()
                print(f"Downloaded model to: {model_dir}")

            # Create output directory
            output_dir = Path("artifacts") / "training" / generate_folder_name()
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_dir}")

            # Log configuration file to artifact
            config_path = Path(config)
            if config_path.exists():
                artifact.add_file(str(config_path), name="training_config.yaml")

            if dry_run:
                print("\n🏃 Dry run mode - skipping training")
                print("Configuration validation and setup completed successfully!")
                return

            # Initialize trainer
            print("\n🚀 Initializing trainer...")
            trainer = NMTTrainer(
                config=training_config,
                artifact=artifact,
                input_data_dir=splits_dir,
                output_data_dir=str(output_dir),
                model_dir=model_dir
            )

            # Initialize S3 loader
            s3_loader = get_s3_loader(bucket)

            # Run training pipeline
            print("\n🎯 Starting training pipeline...")
            final_stats = trainer.run_full_training()

            # Save artifacts and upload to S3
            print("\n💾 Saving artifacts...")

            # Upload to S3
            s3_loader.upload(output_dir)
            artifact.add_reference(f"s3://{bucket}/{str(output_dir)}")

            # Log artifact
            run.log_artifact(artifact)

            # Print final statistics
            print("\n📈 Training completed successfully!")
            print("Final statistics:")
            for key, value in final_stats.items():
                if isinstance(value, (int, float, str)):
                    print(f"  {key}: {value}")

            print(f"\n📁 Results saved to: {output_dir}")
            print(f"☁️  Uploaded to: s3://{bucket}/{output_dir}")
            print(f"🌐 WandB run: {run.url}")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise


@click.command()
@click.option(
    "--registry",
    type=click.Choice(['models', 'trainers', 'evaluators', 'all']),
    default='all',
    help="Which registry to list components from"
)
def list_components(registry: str) -> None:
    """List available components in the registries."""
    from .registry import model_registry, trainer_registry, evaluator_registry

    print("Available Training Components")
    print("=" * 50)

    if registry in ['models', 'all']:
        print("\n🔧 MODELS:")
        for name, desc in model_registry.list_components().items():
            print(f"  {name:<25} - {desc}")

    if registry in ['trainers', 'all']:
        print("\n🏋️ TRAINERS:")
        for name, desc in trainer_registry.list_components().items():
            print(f"  {name:<25} - {desc}")

    if registry in ['evaluators', 'all']:
        print("\n📊 EVALUATORS:")
        for name, desc in evaluator_registry.list_components().items():
            print(f"  {name:<25} - {desc}")


# Create CLI group
@click.group()
def cli():
    """Neural Machine Translation Training Pipeline"""
    pass


# Add commands to group
cli.add_command(main, name="train")
cli.add_command(list_components, name="list")

if __name__ == '__main__':
    cli()
