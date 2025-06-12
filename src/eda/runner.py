#!/usr/bin/env python3
"""
EDA Runner Script

CLI tool for running comprehensive EDA on multilingual datasets from wandb artifacts.
This script is designed to be run outside the main data pipeline.

Usage:
    python eda_runner.py --artifact "raw:v1" --project "NMT_EDA" --limit 1000
"""

import os
from pathlib import Path

import click
import wandb
from dotenv import load_dotenv

from src.eda.analysis import MultilingualEDA

# Load environment variables
load_dotenv('.env')


@click.command()
@click.option(
    "--artifact",
    required=True,
    type=str,
    help="wandb artifact name (e.g., 'raw:v1', 'username/project/artifact:version')"
)
@click.option(
    "--project",
    default="NMT_Training",
    type=str,
    help="wandb project name for EDA results"
)
@click.option(
    "--limit",
    default=-1,
    type=int,
    help="Maximum number of samples to analyze (-1 for all)"
)
@click.option(
    "--run-name",
    default=None,
    type=str,
    help="Optional name for the wandb run"
)
@click.option(
    "--develop",
    is_flag=True,
    default=False,
    help="Run in development mode without logging to personal account"
)
@click.option(
    "--analysis-type",
    default="complete",
    type=click.Choice(['complete', 'basic', 'text', 'quality', 'clustering']),
    help="Type of analysis to run"
)
def main(
        artifact: str,
        project: str,
        limit: int,
        run_name: str,
        develop: bool,
        analysis_type: str
) -> None:
    """
    Run comprehensive EDA on multilingual datasets from wandb artifacts.

    This tool downloads a wandb artifact containing multilingual datasets
    and performs extensive exploratory data analysis, logging all results
    back to wandb for easy sharing and reproducibility.

    Examples:
        # Run complete analysis on latest version
        python eda_runner.py --artifact "raw:latest"

        # Run on specific version with sample limit
        python eda_runner.py --artifact "raw:v3" --limit 5000

        # Run in development mode
        python eda_runner.py --artifact "raw:v1" --develop

        # Run only text analysis
        python eda_runner.py --artifact "raw:v1" --analysis-type text
    """

    # Remove API key if in develop mode
    if develop:
        os.environ.pop('WANDB_API_KEY', None)
        print("Running in development mode (offline)")

    # Print configuration
    print("=== EDA Configuration ===")
    print(f"Artifact: {artifact}")
    print(f"Project: {project}")
    print(f"Sample limit: {limit if limit != -1 else 'All samples'}")
    print(f"Run name: {run_name or 'Auto-generated'}")
    print(f"Analysis type: {analysis_type}")
    print(f"Development mode: {develop}")
    print("=" * 25)

    try:
        # Initialize EDA
        print(f"\nInitializing EDA for artifact: {artifact}")
        eda = MultilingualEDA(
            artifact_name=artifact,
            limit=limit,
            project=project,
            run_name=run_name
        )

        # Run selected analysis
        if analysis_type == "complete":
            print("Running complete EDA analysis...")
            eda.run_complete_analysis()

        elif analysis_type == "basic":
            print("Running basic analysis...")
            eda.basic_statistics()
            eda.domain_analysis()
            eda.quality_analysis()

        elif analysis_type == "text":
            print("Running text analysis...")
            eda.text_length_analysis()
            eda.longest_words_analysis()
            eda.character_analysis()
            eda.word_analysis()
            eda.zipf_analysis()

        elif analysis_type == "quality":
            print("Running quality analysis...")
            eda.basic_statistics()
            eda.quality_analysis()
            eda.language_analysis()

        elif analysis_type == "clustering":
            print("Running clustering analysis...")
            eda.basic_statistics()
            eda.clustering_analysis()

        print("\n✅ EDA completed successfully!")
        print(f"Results logged to wandb project: {project}")

        if eda.run:
            print(f"Run URL: {eda.run.url}")

    except Exception as e:
        print(f"\n❌ EDA failed with error: {str(e)}")
        raise

    finally:
        # Cleanup
        if 'eda' in locals():
            del eda


@click.command()
@click.option(
    "--project",
    default="NMT_Training",
    type=str,
    help="wandb project to list artifacts from"
)
@click.option(
    "--entity",
    default=None,
    type=str,
    help="wandb entity (username/team)"
)
def list_artifacts(project: str, entity: str) -> None:
    """List available artifacts in a wandb project."""

    try:
        # Initialize wandb API
        api = wandb.Api()

        # Get project
        if entity:
            project_path = f"{entity}/{project}"
        else:
            project_path = project

        print(f"Listing artifacts from project: {project_path}")

        # Get artifacts
        artifacts = api.artifacts(project_path, per_page=50)

        print("\nAvailable artifacts:")
        print("-" * 60)
        print(f"{'Name':<20} {'Version':<10} {'Type':<15} {'Size':<10}")
        print("-" * 60)

        for artifact in artifacts:
            size_mb = artifact.size / (1024 * 1024) if artifact.size else 0
            print(f"{artifact.name:<20} {artifact.version:<10} {artifact.type:<15} {size_mb:.1f}MB")

    except Exception as e:
        print(f"Error listing artifacts: {str(e)}")
        print("Make sure you're logged into wandb and have access to the project.")


# Create a CLI group with multiple commands
@click.group()
def cli():
    """Multilingual Dataset EDA Tool

    A comprehensive tool for performing exploratory data analysis
    on multilingual datasets stored as wandb artifacts.
    """
    pass


# Add commands to the group
cli.add_command(main, name="run")
cli.add_command(list_artifacts, name="list")

if __name__ == "__main__":
    cli()
