import os
from tqdm import tqdm
from pathlib import Path
import click
import wandb

from src.loader.loader import DataLoader
from src.utils.utils import generate_folder_name, get_s3_loader

tqdm.pandas()


@click.command()
@click.option(
    "--bucket",
    default="personal-data-science-data",
    required=False,
    type=str,
    help="bucket name where all the data is saved",
)
@click.option(
    "--project",
    default="NMT_Training",
    required=False,
    type=str,
    help="project name to log on wandb",
)
@click.option(
    "--description",
    required=True,
    type=str,
    help="experiment description",
)
def main(
        bucket: str,
        project: str,
        description: str,
        develop: bool
) -> None:
    """
    Main function to run the data loading pipeline.

    Args:
        bucket (str): bucket name where all the data is saved
        project (str): The name of the wandb project.
        description (str): A description of the current run.
        develop (bool): If True, run in development mode without logging to personal account.
    """

    # remove api key if its in develop mode
    if develop:
        os.environ.pop('WANDB_API_KEY', None)

    # Print all arguments for logging purposes
    for arg, value in locals().items():
        print(f"{arg}: {value}")

    # Initialize wandb run
    with wandb.init(project=project,
                    job_type="load-data",
                    tags=["pipeline", "raw"],
                    notes=description,
                    save_code=True,
                    anonymous="must" if develop else "allow") as run:
        # Create a new artifact
        artifact = wandb.Artifact(name="raw", type="dataset", description=description)

        # ------------------ create necessary folders ------------------
        output_folder_dir = Path('artifacts') / "raw" / generate_folder_name()
        output_folder_dir.mkdir(parents=True, exist_ok=True)
        artifact.metadata.update({'output_dir': output_folder_dir})

        # ------------------ load data ------------------
        s3_loader = get_s3_loader(bucket)
        loader = DataLoader(artifact, str(output_folder_dir))
        loader.run_loading()

        # ------------------ save ------------------
        s3_loader.upload(output_folder_dir)
        artifact.add_reference(f"s3://{bucket}/{str(output_folder_dir)}")
        run.log_artifact(artifact)


if __name__ == '__main__':
    main()