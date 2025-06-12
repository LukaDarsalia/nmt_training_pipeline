"""
Exploratory Data Analysis Module for Multilingual Datasets

This module provides comprehensive EDA capabilities for datasets loaded through
the data pipeline, with automatic wandb logging for reproducibility.
"""

import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import wandb
from nltk import ngrams, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.utils.multilingual_dataset import MultilingualDataset

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MultilingualEDA:
    """
    Comprehensive EDA class for multilingual datasets with wandb integration.

    This class provides extensive analysis capabilities for Georgian-English
    parallel datasets and automatically logs all results to wandb.
    """

    def __init__(
            self,
            artifact_name: str,
            limit: int = -1,
            project: str = "NMT_EDA",
            word_tokenizer=lambda x: x.strip().split(' '),
            run_name: Optional[str] = None
    ):
        """
        Initialize the EDA class with wandb artifact.

        Args:
            artifact_name: Name of the wandb artifact (e.g., "raw:v1")
            limit: Maximum number of samples to analyze (-1 for all)
            project: wandb project name
            word_tokenizer: Function to tokenize words
            run_name: Optional name for the wandb run
        """
        self.limit = limit
        self.word_tokenizer = word_tokenizer
        self.artifact_name = artifact_name

        # Initialize wandb
        self.run = wandb.init(
            project=project,
            job_type="eda",
            tags=["analysis", "eda", "multilingual"],
            name=run_name or f"eda_{artifact_name.replace(':', '_')}",
            save_code=True
        )

        # Download and load data
        self.data_dir = self._download_artifact()
        self.combined_df = self._load_all_datasets()

        if limit != -1 and limit < len(self.combined_df):
            self.combined_df = self.combined_df.sample(n=limit, random_state=42)

        # Create datasets for analysis
        self.multilingual_dataset = MultilingualDataset(self.combined_df)

        # Analysis patterns
        self.patterns = {
            r'\d': "one_digit",
            r'\d{2}': "two_digits",
            r'\d{3}': "three_digits",
            r'\d{4}': "four_digits",
        }

        print(f"Initialized EDA with {len(self.combined_df)} samples")
        print(f"Available domains: {self.multilingual_dataset.get_domains()}")
        print(f"Available dataset names: {self.multilingual_dataset.get_dataset_names()}")

    def _download_artifact(self) -> Path:
        """Download wandb artifact and return path to data directory."""
        artifact = self.run.use_artifact(self.artifact_name)
        artifact_dir = Path(artifact.download())
        return artifact_dir

    def _load_all_datasets(self) -> pd.DataFrame:
        """Load all parquet files from the artifact directory."""
        data_frames = []

        for parquet_file in self.data_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            df['source_dataset'] = parquet_file.stem
            data_frames.append(df)

        if not data_frames:
            raise ValueError("No parquet files found in artifact directory")

        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df

    def run_complete_analysis(self) -> None:
        """Run all EDA analyses and log results to wandb."""
        print("Starting comprehensive EDA analysis...")

        # Basic statistics
        self.basic_statistics()

        # Text length analysis
        self.text_length_analysis()

        # Language analysis
        self.language_analysis()

        # Character and word analysis
        self.character_analysis()
        self.word_analysis()

        # Domain and dataset analysis
        self.domain_analysis()

        # N-gram analysis
        self.ngram_analysis()

        # Advanced text analysis
        self.sentence_length_analysis()
        self.word_patterns_analysis()
        self.longest_words_analysis()

        # Word cloud generation
        self.word_cloud_analysis()

        # Clustering analysis
        self.clustering_analysis()

        # Quality checks
        self.quality_analysis()

        print("EDA analysis completed!")

    def basic_statistics(self) -> None:
        """Analyze basic dataset statistics."""
        print("Analyzing basic statistics...")

        stats = {
            'total_samples': len(self.combined_df),
            'unique_domains': len(self.combined_df['domain'].unique()),
            'unique_datasets': len(self.combined_df['source_dataset'].unique()),
            'missing_titles': self.combined_df['title'].isnull().sum(),
            'empty_georgian': (self.combined_df['ka'] == '').sum(),
            'empty_english': (self.combined_df['en'] == '').sum(),
        }

        # Log basic stats
        wandb.log(stats)

        # Create summary table
        summary_df = pd.DataFrame([stats]).T
        summary_df.columns = ['Count']
        summary_table = wandb.Table(dataframe=summary_df.reset_index())
        wandb.log({"basic_statistics": summary_table})

        print(f"Basic statistics logged to wandb")

    def text_length_analysis(self) -> None:
        """Analyze text lengths for both languages."""
        print("Analyzing text lengths...")

        # Calculate lengths
        self.combined_df['ka_char_count'] = self.combined_df['ka'].str.len()
        self.combined_df['en_char_count'] = self.combined_df['en'].str.len()
        self.combined_df['ka_word_count'] = self.combined_df['ka'].apply(
            lambda x: len(self.word_tokenizer(str(x)))
        )
        self.combined_df['en_word_count'] = self.combined_df['en'].apply(
            lambda x: len(self.word_tokenizer(str(x)))
        )

        # Statistics
        length_stats = {}
        for lang in ['ka', 'en']:
            for metric in ['char_count', 'word_count']:
                col = f'{lang}_{metric}'
                length_stats.update({
                    f'{col}_mean': self.combined_df[col].mean(),
                    f'{col}_median': self.combined_df[col].median(),
                    f'{col}_std': self.combined_df[col].std(),
                    f'{col}_min': self.combined_df[col].min(),
                    f'{col}_max': self.combined_df[col].max(),
                })

        wandb.log(length_stats)

        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Character count distributions
        axes[0, 0].hist(self.combined_df['ka_char_count'], bins=50, alpha=0.7, label='Georgian')
        axes[0, 0].hist(self.combined_df['en_char_count'], bins=50, alpha=0.7, label='English')
        axes[0, 0].set_title('Character Count Distribution')
        axes[0, 0].legend()

        # Word count distributions
        axes[0, 1].hist(self.combined_df['ka_word_count'], bins=50, alpha=0.7, label='Georgian')
        axes[0, 1].hist(self.combined_df['en_word_count'], bins=50, alpha=0.7, label='English')
        axes[0, 1].set_title('Word Count Distribution')
        axes[0, 1].legend()

        # Length ratio analysis
        self.combined_df['char_ratio'] = self.combined_df['ka_char_count'] / (self.combined_df['en_char_count'] + 1e-6)
        self.combined_df['word_ratio'] = self.combined_df['ka_word_count'] / (self.combined_df['en_word_count'] + 1e-6)

        axes[1, 0].hist(self.combined_df['char_ratio'], bins=50)
        axes[1, 0].set_title('Character Length Ratio (Georgian/English)')

        axes[1, 1].hist(self.combined_df['word_ratio'], bins=50)
        axes[1, 1].set_title('Word Length Ratio (Georgian/English)')

        plt.tight_layout()
        wandb.log({"text_length_analysis": wandb.Image(fig)})
        plt.close()

    def language_analysis(self) -> None:
        """Analyze language detection and purity."""
        print("Analyzing language detection...")

        # Download FastText model if not exists
        model_path = 'lid.176.ftz'
        if not os.path.exists(model_path):
            print("Downloading FastText language detection model...")
            model_url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'
            response = requests.get(model_url)
            with open(model_path, 'wb') as file:
                file.write(response.content)

        fasttext_model = fasttext.load_model(model_path)

        def detect_language(text: str) -> Tuple[str, float]:
            try:
                predictions = fasttext_model.predict(str(text), k=1)
                lang = predictions[0][0].replace('__label__', '')
                confidence = predictions[1][0]
                return lang, confidence
            except:
                return "unknown", 0.0

        # Detect languages
        print("Detecting Georgian text languages...")
        ka_lang_results = [detect_language(text) for text in tqdm(self.combined_df['ka'])]

        print("Detecting English text languages...")
        en_lang_results = [detect_language(text) for text in tqdm(self.combined_df['en'])]

        # Extract results
        self.combined_df['ka_detected_lang'] = [r[0] for r in ka_lang_results]
        self.combined_df['ka_lang_confidence'] = [r[1] for r in ka_lang_results]
        self.combined_df['en_detected_lang'] = [r[0] for r in en_lang_results]
        self.combined_df['en_lang_confidence'] = [r[1] for r in en_lang_results]

        # Language purity metrics
        ka_purity = (self.combined_df['ka_detected_lang'] == 'ka').mean()
        en_purity = (self.combined_df['en_detected_lang'] == 'en').mean()

        lang_stats = {
            'georgian_purity': ka_purity,
            'english_purity': en_purity,
            'avg_ka_confidence': self.combined_df['ka_lang_confidence'].mean(),
            'avg_en_confidence': self.combined_df['en_lang_confidence'].mean(),
        }

        wandb.log(lang_stats)

        # Create language distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Georgian language distribution
        ka_lang_counts = self.combined_df['ka_detected_lang'].value_counts().head(10)
        axes[0, 0].bar(ka_lang_counts.index, ka_lang_counts.values)
        axes[0, 0].set_title('Detected Languages in Georgian Text')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # English language distribution
        en_lang_counts = self.combined_df['en_detected_lang'].value_counts().head(10)
        axes[0, 1].bar(en_lang_counts.index, en_lang_counts.values)
        axes[0, 1].set_title('Detected Languages in English Text')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Confidence distributions
        axes[1, 0].hist(self.combined_df['ka_lang_confidence'], bins=50)
        axes[1, 0].set_title('Georgian Language Detection Confidence')

        axes[1, 1].hist(self.combined_df['en_lang_confidence'], bins=50)
        axes[1, 1].set_title('English Language Detection Confidence')

        plt.tight_layout()
        wandb.log({"language_analysis": wandb.Image(fig)})
        plt.close()

    def character_analysis(self) -> None:
        """Analyze character frequencies and special characters."""
        print("Analyzing character frequencies...")

        # Character frequency analysis for both languages
        ka_chars = Counter()
        en_chars = Counter()

        for text in tqdm(self.combined_df['ka'], desc="Analyzing Georgian characters"):
            ka_chars.update(str(text))

        for text in tqdm(self.combined_df['en'], desc="Analyzing English characters"):
            en_chars.update(str(text))

        # Get top characters
        top_ka_chars = ka_chars.most_common(50)
        top_en_chars = en_chars.most_common(50)

        # Special characters analysis
        special_chars_ka = {char: count for char, count in top_ka_chars if char in string.punctuation}
        special_chars_en = {char: count for char, count in top_en_chars if char in string.punctuation}

        # Log character statistics
        char_stats = {
            'unique_ka_chars': len(ka_chars),
            'unique_en_chars': len(en_chars),
            'total_ka_chars': sum(ka_chars.values()),
            'total_en_chars': sum(en_chars.values()),
        }

        wandb.log(char_stats)

        # Create character frequency tables
        ka_char_df = pd.DataFrame(top_ka_chars, columns=['Character', 'Frequency'])
        en_char_df = pd.DataFrame(top_en_chars, columns=['Character', 'Frequency'])

        wandb.log({
            "top_georgian_characters": wandb.Table(dataframe=ka_char_df),
            "top_english_characters": wandb.Table(dataframe=en_char_df)
        })

    def word_analysis(self) -> None:
        """Analyze word frequencies, lengths, and patterns."""
        print("Analyzing word patterns...")

        # Word frequency analysis
        ka_words = Counter()
        en_words = Counter()

        for text in tqdm(self.combined_df['ka'], desc="Analyzing Georgian words"):
            ka_words.update(self.word_tokenizer(str(text)))

        for text in tqdm(self.combined_df['en'], desc="Analyzing English words"):
            en_words.update(self.word_tokenizer(str(text)))

        # Word statistics
        word_stats = {
            'unique_ka_words': len(ka_words),
            'unique_en_words': len(en_words),
            'total_ka_words': sum(ka_words.values()),
            'total_en_words': sum(en_words.values()),
        }

        wandb.log(word_stats)

        # Top words
        top_ka_words = ka_words.most_common(100)
        top_en_words = en_words.most_common(100)

        ka_words_df = pd.DataFrame(top_ka_words, columns=['Word', 'Frequency'])
        en_words_df = pd.DataFrame(top_en_words, columns=['Word', 'Frequency'])

        wandb.log({
            "top_georgian_words": wandb.Table(dataframe=ka_words_df),
            "top_english_words": wandb.Table(dataframe=en_words_df)
        })

        # Word length analysis
        ka_word_lengths = []
        en_word_lengths = []

        for words in [self.word_tokenizer(str(text)) for text in self.combined_df['ka']]:
            ka_word_lengths.extend([len(word) for word in words])

        for words in [self.word_tokenizer(str(text)) for text in self.combined_df['en']]:
            en_word_lengths.extend([len(word) for word in words])

        # Plot word length distributions
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].hist(ka_word_lengths, bins=50, alpha=0.7)
        axes[0].set_title('Georgian Word Length Distribution')
        axes[0].set_xlabel('Word Length')
        axes[0].set_ylabel('Frequency')

        axes[1].hist(en_word_lengths, bins=50, alpha=0.7)
        axes[1].set_title('English Word Length Distribution')
        axes[1].set_xlabel('Word Length')
        axes[1].set_ylabel('Frequency')

        plt.tight_layout()
        wandb.log({"word_length_analysis": wandb.Image(fig)})
        plt.close()

    def domain_analysis(self) -> None:
        """Analyze domain and dataset distribution."""
        print("Analyzing domain distribution...")

        # Domain statistics
        domain_counts = self.combined_df['domain'].value_counts()
        dataset_counts = self.combined_df['source_dataset'].value_counts()

        # Log domain/dataset statistics
        domain_stats = {
            'num_domains': len(domain_counts),
            'num_datasets': len(dataset_counts),
            'largest_domain_size': domain_counts.iloc[0],
            'smallest_domain_size': domain_counts.iloc[-1],
        }

        wandb.log(domain_stats)

        # Create distribution plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Domain distribution
        domain_counts.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Samples per Domain')
        axes[0].tick_params(axis='x', rotation=45)

        # Dataset distribution
        dataset_counts.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Samples per Source Dataset')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        wandb.log({"domain_analysis": wandb.Image(fig)})
        plt.close()

        # Log detailed tables
        domain_df = domain_counts.reset_index()
        domain_df.columns = ['Domain', 'Count']

        dataset_df = dataset_counts.reset_index()
        dataset_df.columns = ['Dataset', 'Count']

        wandb.log({
            "domain_distribution": wandb.Table(dataframe=domain_df),
            "dataset_distribution": wandb.Table(dataframe=dataset_df)
        })

    def word_cloud_analysis(self) -> None:
        """Generate and log word clouds for both languages."""
        print("Generating word clouds...")

        try:
            from wordcloud import WordCloud
        except ImportError:
            print("WordCloud not installed. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "wordcloud"])
            from wordcloud import WordCloud

        # Combine all text for each language
        ka_text = ' '.join(self.combined_df['ka'].astype(str))
        en_text = ' '.join(self.combined_df['en'].astype(str))

        # Create word clouds
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Georgian word cloud
        ka_wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            font_path='./bpg_dejavu_sans.otf',
            colormap='viridis'
        ).generate(ka_text)

        axes[0].imshow(ka_wordcloud, interpolation='bilinear')
        axes[0].set_title('Georgian Text Word Cloud', fontsize=16)
        axes[0].axis('off')

        # English word cloud
        en_wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            font_path='./bpg_dejavu_sans.otf',
            colormap='plasma'
        ).generate(en_text)

        axes[1].imshow(en_wordcloud, interpolation='bilinear')
        axes[1].set_title('English Text Word Cloud', fontsize=16)
        axes[1].axis('off')

        plt.tight_layout()
        wandb.log({"word_clouds": wandb.Image(fig)})
        plt.close()

        # Domain-specific word clouds
        self._generate_domain_word_clouds()

    def _generate_domain_word_clouds(self) -> None:
        """Generate word clouds for each domain."""
        try:
            from wordcloud import WordCloud
        except ImportError:
            return

        domains = self.combined_df['domain'].unique()[:6]  # Limit to 6 domains for visualization

        if len(domains) > 1:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, domain in enumerate(domains):
                if i >= 6:  # Maximum 6 subplots
                    break

                domain_data = self.combined_df[self.combined_df['domain'] == domain]
                combined_text = ' '.join(domain_data['ka'].astype(str) + ' ' + domain_data['en'].astype(str))

                if len(combined_text.strip()) > 0:
                    wordcloud = WordCloud(
                        width=400,
                        height=300,
                        background_color='white',
                        max_words=50,
                        font_path='./bpg_dejavu_sans.otf',
                        colormap='Set3'
                    ).generate(combined_text)

                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(f'Domain: {domain}', fontsize=12)
                    axes[i].axis('off')
                else:
                    axes[i].axis('off')

            # Hide unused subplots
            for i in range(len(domains), 6):
                axes[i].axis('off')

            plt.tight_layout()
            wandb.log({"domain_word_clouds": wandb.Image(fig)})
            plt.close()

    def ngram_analysis(self, n_values: List[int] = [2, 3, 4]) -> None:
        """Analyze n-gram frequencies."""
        print("Analyzing n-grams...")

        for n in n_values:
            print(f"Analyzing {n}-grams...")

            # Georgian n-grams
            ka_ngrams = Counter()
            for text in tqdm(self.combined_df['ka'], desc=f"Georgian {n}-grams"):
                words = self.word_tokenizer(str(text))
                if len(words) >= n:
                    ka_ngrams.update([' '.join(ngram) for ngram in ngrams(words, n)])

            # English n-grams
            en_ngrams = Counter()
            for text in tqdm(self.combined_df['en'], desc=f"English {n}-grams"):
                words = self.word_tokenizer(str(text))
                if len(words) >= n:
                    en_ngrams.update([' '.join(ngram) for ngram in ngrams(words, n)])

            # Log top n-grams
            top_ka_ngrams = ka_ngrams.most_common(50)
            top_en_ngrams = en_ngrams.most_common(50)

            ka_ngrams_df = pd.DataFrame(top_ka_ngrams, columns=[f'{n}-gram', 'Frequency'])
            en_ngrams_df = pd.DataFrame(top_en_ngrams, columns=[f'{n}-gram', 'Frequency'])

    def ngram_analysis(self, n_values: List[int] = [2, 3, 4]) -> None:
        """Analyze n-gram frequencies with enhanced visualization."""
        print("Analyzing n-grams...")

        all_ngram_data = {}

        for n in n_values:
            print(f"Analyzing {n}-grams...")

            # Georgian n-grams
            ka_ngrams = Counter()
            for text in tqdm(self.combined_df['ka'], desc=f"Georgian {n}-grams"):
                words = self.word_tokenizer(str(text))
                if len(words) >= n:
                    ka_ngrams.update([' '.join(ngram) for ngram in ngrams(words, n)])

            # English n-grams
            en_ngrams = Counter()
            for text in tqdm(self.combined_df['en'], desc=f"English {n}-grams"):
                words = self.word_tokenizer(str(text))
                if len(words) >= n:
                    en_ngrams.update([' '.join(ngram) for ngram in ngrams(words, n)])

            # Store for visualization
            all_ngram_data[n] = {
                'ka': ka_ngrams.most_common(20),
                'en': en_ngrams.most_common(20)
            }

            # Log top n-grams as tables
            top_ka_ngrams = ka_ngrams.most_common(50)
            top_en_ngrams = en_ngrams.most_common(50)

            ka_ngrams_df = pd.DataFrame(top_ka_ngrams, columns=[f'{n}-gram', 'Frequency'])
            en_ngrams_df = pd.DataFrame(top_en_ngrams, columns=[f'{n}-gram', 'Frequency'])

            wandb.log({
                f"top_georgian_{n}grams": wandb.Table(dataframe=ka_ngrams_df),
                f"top_english_{n}grams": wandb.Table(dataframe=en_ngrams_df)
            })

            # Log n-gram statistics
            ngram_stats = {
                f'unique_ka_{n}grams': len(ka_ngrams),
                f'unique_en_{n}grams': len(en_ngrams),
                f'total_ka_{n}grams': sum(ka_ngrams.values()),
                f'total_en_{n}grams': sum(en_ngrams.values()),
            }
            wandb.log(ngram_stats)

        # Create comprehensive n-gram visualization
        self._plot_ngram_comparison(all_ngram_data)

    def _plot_ngram_comparison(self, ngram_data: Dict) -> None:
        """Create comparison plots for n-grams."""
        n_values = list(ngram_data.keys())

        fig, axes = plt.subplots(len(n_values), 2, figsize=(20, 6 * len(n_values)))
        if len(n_values) == 1:
            axes = axes.reshape(1, -1)

        for i, n in enumerate(n_values):
            # Georgian n-grams
            ka_ngrams = ngram_data[n]['ka'][:10]  # Top 10 for visualization
            if ka_ngrams:
                y_pos = np.arange(len(ka_ngrams))
                frequencies = [freq for _, freq in ka_ngrams]
                labels = [ngram for ngram, _ in ka_ngrams]

                axes[i, 0].barh(y_pos, frequencies)
                axes[i, 0].set_yticks(y_pos)
                axes[i, 0].set_yticklabels(labels, fontsize=8)
                axes[i, 0].set_title(f'Top Georgian {n}-grams')
                axes[i, 0].set_xlabel('Frequency')

            # English n-grams
            en_ngrams = ngram_data[n]['en'][:10]  # Top 10 for visualization
            if en_ngrams:
                y_pos = np.arange(len(en_ngrams))
                frequencies = [freq for _, freq in en_ngrams]
                labels = [ngram for ngram, _ in en_ngrams]

                axes[i, 1].barh(y_pos, frequencies)
                axes[i, 1].set_yticks(y_pos)
                axes[i, 1].set_yticklabels(labels, fontsize=8)
                axes[i, 1].set_title(f'Top English {n}-grams')
                axes[i, 1].set_xlabel('Frequency')

        plt.tight_layout()
        wandb.log({"ngram_comparison": wandb.Image(fig)})
        plt.close()

    def sentence_length_analysis(self) -> None:
        """Analyze sentence length distributions."""
        print("Analyzing sentence lengths...")

        # Calculate sentence lengths
        ka_sentence_lengths = []
        en_sentence_lengths = []

        for text in tqdm(self.combined_df['ka'], desc="Analyzing Georgian sentences"):
            try:
                sentences = sent_tokenize(str(text))
                ka_sentence_lengths.extend([len(self.word_tokenizer(sent)) for sent in sentences])
            except:
                # Fallback if sent_tokenize fails
                ka_sentence_lengths.append(len(self.word_tokenizer(str(text))))

        for text in tqdm(self.combined_df['en'], desc="Analyzing English sentences"):
            try:
                sentences = sent_tokenize(str(text))
                en_sentence_lengths.extend([len(self.word_tokenizer(sent)) for sent in sentences])
            except:
                # Fallback if sent_tokenize fails
                en_sentence_lengths.append(len(self.word_tokenizer(str(text))))

        # Log statistics
        sentence_stats = {
            'avg_ka_sentence_length': np.mean(ka_sentence_lengths),
            'avg_en_sentence_length': np.mean(en_sentence_lengths),
            'median_ka_sentence_length': np.median(ka_sentence_lengths),
            'median_en_sentence_length': np.median(en_sentence_lengths),
            'max_ka_sentence_length': np.max(ka_sentence_lengths),
            'max_en_sentence_length': np.max(en_sentence_lengths),
        }
        wandb.log(sentence_stats)

        # Plot sentence length distributions
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].hist(ka_sentence_lengths, bins=50, alpha=0.7, color='blue')
        axes[0].set_title('Georgian Sentence Length Distribution')
        axes[0].set_xlabel('Sentence Length (words)')
        axes[0].set_ylabel('Frequency')

        axes[1].hist(en_sentence_lengths, bins=50, alpha=0.7, color='orange')
        axes[1].set_title('English Sentence Length Distribution')
        axes[1].set_xlabel('Sentence Length (words)')
        axes[1].set_ylabel('Frequency')

        plt.tight_layout()
        wandb.log({"sentence_length_analysis": wandb.Image(fig)})
        plt.close()

    def word_patterns_analysis(self) -> None:
        """Analyze word patterns including beginnings, endings, and number patterns."""
        print("Analyzing word patterns...")

        # Word beginnings and endings
        ka_beginnings = Counter()
        ka_endings = Counter()
        en_beginnings = Counter()
        en_endings = Counter()

        for text in tqdm(self.combined_df['ka'], desc="Georgian word patterns"):
            words = self.word_tokenizer(str(text))
            for word in words:
                if len(word) > 0:
                    ka_beginnings[word[0]] += 1
                    ka_endings[word[-1]] += 1

        for text in tqdm(self.combined_df['en'], desc="English word patterns"):
            words = self.word_tokenizer(str(text))
            for word in words:
                if len(word) > 0:
                    en_beginnings[word[0]] += 1
                    en_endings[word[-1]] += 1

        # Number patterns
        number_patterns = {
            r'\d': "single_digits",
            r'\d{2}': "two_digits",
            r'\d{3}': "three_digits",
            r'\d{4}': "four_digits",
            r'\d{4,}': "long_numbers"
        }

        pattern_counts = {pattern_name: 0 for pattern_name in number_patterns.values()}

        all_text = ' '.join(self.combined_df['ka'].astype(str) + ' ' + self.combined_df['en'].astype(str))
        for pattern, pattern_name in number_patterns.items():
            pattern_counts[pattern_name] = len(re.findall(pattern, all_text))

        # Log pattern statistics
        wandb.log(pattern_counts)

        # Create word pattern tables
        ka_beginnings_df = pd.DataFrame(ka_beginnings.most_common(20), columns=['Character', 'Frequency'])
        ka_endings_df = pd.DataFrame(ka_endings.most_common(20), columns=['Character', 'Frequency'])
        en_beginnings_df = pd.DataFrame(en_beginnings.most_common(20), columns=['Character', 'Frequency'])
        en_endings_df = pd.DataFrame(en_endings.most_common(20), columns=['Character', 'Frequency'])

        wandb.log({
            "georgian_word_beginnings": wandb.Table(dataframe=ka_beginnings_df),
            "georgian_word_endings": wandb.Table(dataframe=ka_endings_df),
            "english_word_beginnings": wandb.Table(dataframe=en_beginnings_df),
            "english_word_endings": wandb.Table(dataframe=en_endings_df)
        })

        # Plot number patterns
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        pattern_names = list(pattern_counts.keys())
        pattern_values = list(pattern_counts.values())

        ax.bar(pattern_names, pattern_values)
        ax.set_title('Number Pattern Frequencies')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        wandb.log({"number_patterns": wandb.Image(fig)})
        plt.close()

    def longest_words_analysis(self, n: int = 50) -> None:
        """Analyze longest words in both languages."""
        print("Analyzing longest words...")

        # Collect all words with lengths
        ka_words_with_length = []
        en_words_with_length = []

        for text in tqdm(self.combined_df['ka'], desc="Georgian longest words"):
            words = self.word_tokenizer(str(text))
            ka_words_with_length.extend([(len(word), word) for word in words if len(word) > 0])

        for text in tqdm(self.combined_df['en'], desc="English longest words"):
            words = self.word_tokenizer(str(text))
            en_words_with_length.extend([(len(word), word) for word in words if len(word) > 0])

        # Get longest unique words
        ka_longest = sorted(list(set(ka_words_with_length)), reverse=True)[:n]
        en_longest = sorted(list(set(en_words_with_length)), reverse=True)[:n]

        # Create tables
        ka_longest_df = pd.DataFrame(ka_longest, columns=['Length', 'Word'])
        en_longest_df = pd.DataFrame(en_longest, columns=['Length', 'Word'])

        wandb.log({
            "longest_georgian_words": wandb.Table(dataframe=ka_longest_df),
            "longest_english_words": wandb.Table(dataframe=en_longest_df)
        })

        # Log statistics
        longest_stats = {
            'longest_ka_word_length': ka_longest[0][0] if ka_longest else 0,
            'longest_en_word_length': en_longest[0][0] if en_longest else 0,
            'avg_top50_ka_word_length': np.mean([length for length, _ in ka_longest]),
            'avg_top50_en_word_length': np.mean([length for length, _ in en_longest]),
        }
        wandb.log(longest_stats)

    def clustering_analysis(self, n_components: int = 50, max_clusters: int = 10) -> None:
        """Perform clustering analysis using TF-IDF and LSA."""
        print("Performing clustering analysis...")

        # Combine texts for clustering
        all_texts = (self.combined_df['ka'] + ' ' + self.combined_df['en']).astype(str)

        # TF-IDF Vectorization
        print("Computing TF-IDF...")
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

        # LSA using SVD
        print("Applying LSA...")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        lsa_matrix = svd.fit_transform(tfidf_matrix)

        # Elbow method for optimal clusters
        print("Finding optimal number of clusters...")
        sse = []
        cluster_range = range(2, max_clusters + 1)

        for n_clusters in tqdm(cluster_range):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(lsa_matrix)
            sse.append(kmeans.inertia_)

        # Find elbow point (simplified)
        sse_diff = np.diff(sse)
        elbow_point = np.argmin(sse_diff) + 2  # +2 because we start from 2 clusters

        # Final clustering with optimal number
        print(f"Clustering with {elbow_point} clusters...")
        final_kmeans = KMeans(n_clusters=elbow_point, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(lsa_matrix)

        self.combined_df['cluster'] = cluster_labels

        # Plot elbow curve
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(cluster_range, sse, marker='o')
        axes[0].axvline(x=elbow_point, color='red', linestyle='--')
        axes[0].set_title('Elbow Method for Optimal Clusters')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('SSE')
        axes[0].grid(True)

        # Cluster size distribution
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        axes[1].bar(cluster_counts.index, cluster_counts.values)
        axes[1].set_title('Cluster Size Distribution')
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Number of Samples')

        plt.tight_layout()
        wandb.log({"clustering_analysis": wandb.Image(fig)})
        plt.close()

        # Log cluster samples
        cluster_samples = []
        for cluster_id in range(elbow_point):
            cluster_data = self.combined_df[self.combined_df['cluster'] == cluster_id]
            sample_size = min(5, len(cluster_data))
            samples = cluster_data.sample(sample_size, random_state=42)

            for _, row in samples.iterrows():
                cluster_samples.append({
                    'cluster': cluster_id,
                    'domain': row['domain'],
                    'georgian_text': row['ka'][:100] + '...',
                    'english_text': row['en'][:100] + '...'
                })

        cluster_df = pd.DataFrame(cluster_samples)
        wandb.log({"cluster_samples": wandb.Table(dataframe=cluster_df)})

        # Log clustering metrics
        wandb.log({
            'optimal_clusters': elbow_point,
            'explained_variance_ratio': svd.explained_variance_ratio_.sum(),
        })

    def quality_analysis(self) -> None:
        """Analyze data quality issues."""
        print("Analyzing data quality...")

        # Null and duplicate analysis
        quality_stats = {
            'null_titles': self.combined_df['title'].isnull().sum(),
            'null_georgian': self.combined_df['ka'].isnull().sum(),
            'null_english': self.combined_df['en'].isnull().sum(),
            'empty_georgian': (self.combined_df['ka'] == '').sum(),
            'empty_english': (self.combined_df['en'] == '').sum(),
            'duplicate_pairs': self.combined_df.duplicated(subset=['ka', 'en']).sum(),
            'duplicate_georgian': self.combined_df['ka'].duplicated().sum(),
            'duplicate_english': self.combined_df['en'].duplicated().sum(),
        }

        wandb.log(quality_stats)

        # Length anomalies
        very_short_ka = (self.combined_df['ka_char_count'] < 10).sum()
        very_short_en = (self.combined_df['en_char_count'] < 10).sum()
        very_long_ka = (self.combined_df['ka_char_count'] > 1000).sum()
        very_long_en = (self.combined_df['en_char_count'] > 1000).sum()

        anomaly_stats = {
            'very_short_georgian': very_short_ka,
            'very_short_english': very_short_en,
            'very_long_georgian': very_long_ka,
            'very_long_english': very_long_en,
        }

        wandb.log(anomaly_stats)

        # Create quality summary
        quality_summary = pd.DataFrame([
            {'Issue': 'Null Georgian texts', 'Count': quality_stats['null_georgian']},
            {'Issue': 'Null English texts', 'Count': quality_stats['null_english']},
            {'Issue': 'Empty Georgian texts', 'Count': quality_stats['empty_georgian']},
            {'Issue': 'Empty English texts', 'Count': quality_stats['empty_english']},
            {'Issue': 'Duplicate pairs', 'Count': quality_stats['duplicate_pairs']},
            {'Issue': 'Very short Georgian (<10 chars)', 'Count': very_short_ka},
            {'Issue': 'Very short English (<10 chars)', 'Count': very_short_en},
            {'Issue': 'Very long Georgian (>1000 chars)', 'Count': very_long_ka},
            {'Issue': 'Very long English (>1000 chars)', 'Count': very_long_en},
        ])

        wandb.log({"quality_issues": wandb.Table(dataframe=quality_summary)})

    def zipf_analysis(self, top_n: int = 1000) -> None:
        """Analyze Zipf's law for word frequencies."""
        print("Analyzing Zipf's law...")

        for lang, col in [('Georgian', 'ka'), ('English', 'en')]:
            # Collect all words
            all_words = []
            for text in tqdm(self.combined_df[col], desc=f"Collecting {lang} words"):
                all_words.extend(self.word_tokenizer(str(text)))

            # Count frequencies
            word_counts = Counter(all_words)
            most_common = word_counts.most_common(top_n)

            # Rank-frequency data
            ranks = np.arange(1, len(most_common) + 1)
            frequencies = np.array([freq for _, freq in most_common])

            # Plot Zipf's law
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Linear scale
            axes[0].plot(ranks, frequencies, marker='o', markersize=2)
            axes[0].set_xlabel("Rank")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title(f"Zipf's Law - {lang} (Linear Scale)")
            axes[0].grid(True)

            # Log scale
            axes[1].loglog(ranks, frequencies, marker='o', markersize=2)
            axes[1].set_xlabel("Rank (log)")
            axes[1].set_ylabel("Frequency (log)")
            axes[1].set_title(f"Zipf's Law - {lang} (Log Scale)")
            axes[1].grid(True)

            plt.tight_layout()
            wandb.log({f"zipf_analysis_{lang.lower()}": wandb.Image(fig)})
            plt.close()
