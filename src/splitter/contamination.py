"""
Contamination Detection and Removal

This module provides functions for detecting and removing contaminated samples
between train/valid/test splits to prevent data leakage.
"""

import hashlib
from typing import Dict, List, Tuple, Set, Any

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


class ContaminationDetector:
    """Detects contamination between dataset splits."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize contamination detector.

        Args:
            config: Contamination checking configuration
        """
        self.config = config
        self.methods = config.get('methods', {})
        self.contamination_stats = {}

    def check_contamination(
            self,
            source_df: pd.DataFrame,
            target_df: pd.DataFrame,
            source_name: str,
            target_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Check for contamination between source and target datasets.

        Args:
            source_df: Source DataFrame to check against
            target_df: Target DataFrame to check for contamination
            source_name: Name of source split (e.g., "test")
            target_name: Name of target split (e.g., "train")

        Returns:
            Tuple of (clean_target_df, contaminated_df, stats)
        """
        print(f"\nChecking contamination: {source_name} → {target_name}")

        contaminated_indices = set()
        stats = {
            'source_size': len(source_df),
            'target_size': len(target_df),
            'exact_matches': 0,
            'near_duplicates': 0,
            'total_contaminated': 0
        }

        # Check exact matches
        if self.methods.get('exact_match', {}).get('enabled', True):
            exact_indices, exact_stats = self._check_exact_matches(
                source_df, target_df
            )
            contaminated_indices.update(exact_indices)
            stats['exact_matches'] = exact_stats['num_matches']

        # Check near duplicates
        if self.methods.get('near_duplicate', {}).get('enabled', True):
            near_indices, near_stats = self._check_near_duplicates(
                source_df, target_df
            )
            contaminated_indices.update(near_indices)
            stats['near_duplicates'] = near_stats['num_matches']

        # Get clean and contaminated DataFrames
        contaminated_mask = target_df.index.isin(contaminated_indices)
        clean_df = target_df[~contaminated_mask].copy()
        contaminated_df = target_df[contaminated_mask].copy()

        stats['total_contaminated'] = len(contaminated_df)
        stats['contamination_rate'] = (
            stats['total_contaminated'] / stats['target_size']
            if stats['target_size'] > 0 else 0
        )

        print(f"  Found {stats['total_contaminated']} contaminated samples "
              f"({stats['contamination_rate']:.2%})")
        print(f"    - Exact matches: {stats['exact_matches']}")
        print(f"    - Near duplicates: {stats['near_duplicates']}")

        return clean_df, contaminated_df, stats

    def _check_exact_matches(
            self,
            source_df: pd.DataFrame,
            target_df: pd.DataFrame
    ) -> Tuple[Set[int], Dict[str, Any]]:
        """
        Check for exact matches between source and target.

        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame

        Returns:
            Tuple of (contaminated_indices, stats)
        """
        columns = self.methods['exact_match'].get('columns', ['en', 'ka'])

        # Create hash for each row for efficient comparison
        source_hashes = set()

        print("  Computing source hashes...")
        for _, row in tqdm(source_df.iterrows(), total=len(source_df)):
            row_text = '|'.join([str(row[col]) for col in columns])
            row_hash = hashlib.md5(row_text.encode()).hexdigest()
            source_hashes.add(row_hash)

        # Find matches in target
        contaminated_indices = set()

        print("  Checking target for exact matches...")
        for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
            row_text = '|'.join([str(row[col]) for col in columns])
            row_hash = hashlib.md5(row_text.encode()).hexdigest()

            if row_hash in source_hashes:
                contaminated_indices.add(idx)

        stats = {
            'num_matches': len(contaminated_indices),
            'columns_checked': columns
        }

        return contaminated_indices, stats

    def _check_near_duplicates(
            self,
            source_df: pd.DataFrame,
            target_df: pd.DataFrame
    ) -> Tuple[Set[int], Dict[str, Any]]:
        """
        Check for near duplicates using MinHash LSH.

        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame

        Returns:
            Tuple of (contaminated_indices, stats)
        """
        algorithm = self.methods['near_duplicate'].get('algorithm', 'minhash')

        if algorithm == 'minhash':
            return self._check_minhash_duplicates(source_df, target_df)
        elif algorithm == 'simhash':
            return self._check_simhash_duplicates(source_df, target_df)
        elif algorithm == 'embeddings':
            return self._check_embedding_duplicates(source_df, target_df)
        else:
            raise ValueError(f"Unknown near-duplicate algorithm: {algorithm}")

    def _check_minhash_duplicates(
            self,
            source_df: pd.DataFrame,
            target_df: pd.DataFrame
    ) -> Tuple[Set[int], Dict[str, Any]]:
        """
        Check for near duplicates using MinHash LSH.

        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame

        Returns:
            Tuple of (contaminated_indices, stats)
        """
        params = self.methods['near_duplicate'].get('minhash_params', {})
        num_perm = params.get('num_perm', 128)
        threshold = params.get('threshold', 0.9)
        columns = self.methods['near_duplicate'].get('columns', ['en', 'ka'])

        # Initialize LSH
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

        # Add source samples to LSH
        print("  Building MinHash index for source...")
        for idx, row in tqdm(source_df.iterrows(), total=len(source_df)):
            minhash = MinHash(num_perm=num_perm)

            # Combine columns and create shingles
            text = ' '.join([str(row[col]) for col in columns])
            for word in text.split():
                minhash.update(word.encode('utf8'))

            lsh.insert(f"source_{idx}", minhash)

        # Check target samples
        contaminated_indices = set()

        print("  Checking target for near duplicates...")
        for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
            minhash = MinHash(num_perm=num_perm)

            # Combine columns and create shingles
            text = ' '.join([str(row[col]) for col in columns])
            for word in text.split():
                minhash.update(word.encode('utf8'))

            # Query LSH
            results = lsh.query(minhash)
            if results:
                contaminated_indices.add(idx)

        stats = {
            'num_matches': len(contaminated_indices),
            'algorithm': 'minhash',
            'threshold': threshold,
            'num_perm': num_perm
        }

        return contaminated_indices, stats

    def _check_simhash_duplicates(
            self,
            source_df: pd.DataFrame,
            target_df: pd.DataFrame
    ) -> Tuple[Set[int], Dict[str, Any]]:
        """
        Check for near duplicates using SimHash.
        (Placeholder - implement if needed)
        """
        # For now, return empty results
        return set(), {'num_matches': 0, 'algorithm': 'simhash'}

    def _check_embedding_duplicates(
            self,
            source_df: pd.DataFrame,
            target_df: pd.DataFrame
    ) -> Tuple[Set[int], Dict[str, Any]]:
        """
        Check for near duplicates using embeddings.
        (Placeholder - implement if needed)
        """
        # For now, return empty results
        return set(), {'num_matches': 0, 'algorithm': 'embeddings'}

    def generate_contamination_report(
            self,
            contamination_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate a contamination report DataFrame.

        Args:
            contamination_results: Dictionary of contamination check results

        Returns:
            DataFrame with contamination report
        """
        report_data = []

        for check_pair, results in contamination_results.items():
            source, target = check_pair.split('_to_')

            report_data.append({
                'source_split': source,
                'target_split': target,
                'source_size': results['source_size'],
                'target_size': results['target_size'],
                'exact_matches': results['exact_matches'],
                'near_duplicates': results['near_duplicates'],
                'total_contaminated': results['total_contaminated'],
                'contamination_rate': f"{results['contamination_rate']:.2%}"
            })

        return pd.DataFrame(report_data)
