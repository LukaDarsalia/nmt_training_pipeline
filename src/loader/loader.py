from pathlib import Path

import wandb
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from src.utils.utils import ComposeLoading

tqdm.pandas()


class DataLoader:
    """
    A class for loading and processing various datasets, generating appropriate dataset objects,
    and logging metadata and samples to wandb.
    """
    def __init__(self, artifact: wandb.Artifact, output_folder_dir: str):
        """
        Initialize the DataLoader.

        Args:
            artifact (wandb.Artifact): The wandb artifact to log data to.
            output_folder_dir (str): The directory to save processed datasets.
        """
        self.artifact = artifact
        self.output_folder_dir = output_folder_dir

    def _load_en_ka_corpora(self):
        ds = load_dataset("Darsala/english_georgian_corpora")
        df = ds['train'].to_pandas()
        df.to_parquet(Path(self.output_folder_dir) / "english_georgian_corpora.parquet", index=False)

        sample = df.to_pandas().sample(10, random_state=42)
        table = wandb.Table(dataframe=sample)
        self.artifact.add(table, "english_georgian_corpora Data")

        self.artifact.metadata.update({"english_georgian_corpora Data size": len(df)})

    def _load_flores_devtest(self):
        ds_ka = load_dataset("openlanguagedata/flores_plus", "kat_Geor", split="devtest")
        ds_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")

        ka = ds_ka.to_pandas()
        en = ds_en.to_pandas()

        data = {'title': [], 'ka': [], 'en': [], 'domain': [], 'id': []}

        for i in range(len(en)):
            en_i = en.iloc[i]
            ka_i = ka.iloc[i]

            data['title'].append(None)
            data['ka'].append(ka_i['text'])
            data['en'].append(en_i['text'])
            data['domain'].append(en_i['topic'])
            data['id'].append(f"{en_i['domain']}_{en_i['id']}")

        df = pd.DataFrame(data)
        df.to_parquet(Path(self.output_folder_dir) / "flores_devtest.parquet", index=False)

        sample = df.to_pandas().sample(10, random_state=42)
        table = wandb.Table(dataframe=sample)
        self.artifact.add(table, "flores_devtest Data")

        self.artifact.metadata.update({"flores_devtest Data size": len(df)})

    def load_all_data(self):
        """Load all datasets by calling respective loading methods."""
        loading_functions = [
            self._load_en_ka_corpora,
            self._load_flores_devtest
        ]
        composer = ComposeLoading(loading_functions)
        print(f"Will run {composer}")
        composer()

    def run_loading(self):
        """Main method to initiate the data loading process."""
        self.load_all_data()