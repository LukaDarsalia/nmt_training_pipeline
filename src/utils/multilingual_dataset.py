from typing import Optional, Dict, Any, List

import pandas as pd
import torch
from torch.utils.data import Dataset


class MultilingualDataset(Dataset):
    """
    Custom dataset class for multilingual data with Georgian and English text.

    Expected columns: ['title', 'ka', 'en', 'domain', 'id']
    ID format: "datasetName_some_meta_data"
    """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer=None,
            max_length: int = 512,
            return_tensors: str = "pt",
            include_metadata: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            data: DataFrame with columns ['title', 'ka', 'en', 'domain', 'id']
            tokenizer: Tokenizer for text processing (optional)
            max_length: Maximum sequence length for tokenization
            return_tensors: Format for returned tensors ("pt" for PyTorch)
            include_metadata: Whether to parse and include metadata from ID
        """
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.include_metadata = include_metadata

        # Validate required columns
        required_cols = ['title', 'ka', 'en', 'domain', 'id']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        row = self.data.iloc[idx]

        item = {
            'title': row['title'],
            'ka_text': row['ka'],
            'en_text': row['en'],
            'domain': row['domain'],
            'id': row['id'],
            'index': idx
        }

        # Parse metadata from ID if requested
        if self.include_metadata:
            metadata = self._parse_id_metadata(row['id'])
            item.update(metadata)

        # Tokenize texts if tokenizer is provided
        if self.tokenizer is not None:
            item.update(self._tokenize_texts(row))

        return item

    def _parse_id_metadata(self, id_string: str) -> Dict[str, str]:
        """
        Parse metadata from ID string with format 'datasetName_some_meta_data'.

        Args:
            id_string: ID string to parse

        Returns:
            Dictionary with parsed metadata
        """
        try:
            parts = id_string.split('_', 1)  # Split only on first underscore
            dataset_name = parts[0] if len(parts) > 0 else ""
            metadata = parts[1] if len(parts) > 1 else ""

            return {
                'dataset_name': dataset_name,
                'metadata': metadata,
                'full_id': id_string
            }
        except Exception as e:
            return {
                'dataset_name': "",
                'metadata': "",
                'full_id': id_string,
                'parse_error': str(e)
            }

    def _tokenize_texts(self, row: pd.Series) -> Dict[str, Any]:
        """
        Tokenize the text fields using the provided tokenizer.

        Args:
            row: DataFrame row containing the text data

        Returns:
            Dictionary with tokenized texts
        """
        tokenized = {}

        # Tokenize each text field
        for field, text in [('title', row['title']), ('ka', row['ka']), ('en', row['en'])]:
            if pd.isna(text):
                text = ""

            tokens = self.tokenizer(
                str(text),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors=self.return_tensors
            )

            # Add field prefix to avoid key conflicts
            for key, value in tokens.items():
                tokenized[f'{field}_{key}'] = value.squeeze(0) if self.return_tensors == "pt" else value

        return tokenized

    def get_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get all items from a specific domain."""
        domain_indices = self.data[self.data['domain'] == domain].index.tolist()
        return [self[idx] for idx in domain_indices]

    def get_by_dataset_name(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get all items from a specific dataset (parsed from ID)."""
        if not self.include_metadata:
            raise ValueError("Metadata parsing is disabled. Set include_metadata=True to use this method.")

        items = []
        for idx in range(len(self)):
            item = self[idx]
            if item.get('dataset_name') == dataset_name:
                items.append(item)
        return items

    def get_languages(self) -> List[str]:
        """Get list of available languages."""
        return ['ka', 'en']  # Georgian and English

    def get_domains(self) -> List[str]:
        """Get list of unique domains in the dataset."""
        return self.data['domain'].unique().tolist()

    def get_dataset_names(self) -> List[str]:
        """Get list of unique dataset names (parsed from IDs)."""
        if not self.include_metadata:
            raise ValueError("Metadata parsing is disabled. Set include_metadata=True to use this method.")

        dataset_names = set()
        for idx in range(len(self)):
            item = self[idx]
            if 'dataset_name' in item and item['dataset_name']:
                dataset_names.add(item['dataset_name'])
        return list(dataset_names)

    def filter_by_domain(self, domain: str) -> 'MultilingualDataset':
        """Create a new dataset filtered by domain."""
        filtered_data = self.data[self.data['domain'] == domain].copy()
        return MultilingualDataset(
            filtered_data,
            self.tokenizer,
            self.max_length,
            self.return_tensors,
            self.include_metadata
        )

    def sample(self, n: int, random_state: Optional[int] = None) -> 'MultilingualDataset':
        """Create a new dataset with n random samples."""
        sampled_data = self.data.sample(n=min(n, len(self.data)), random_state=random_state)
        return MultilingualDataset(
            sampled_data,
            self.tokenizer,
            self.max_length,
            self.return_tensors,
            self.include_metadata
        )


# Example usage and utility functions
def create_dataloader(dataset: MultilingualDataset, batch_size: int = 32, shuffle: bool = True):
    """Create a DataLoader for the custom dataset."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_multilingual_batch
    )


def collate_multilingual_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching multilingual data.
    Handles both tokenized and non-tokenized data.
    """
    if not batch:
        return {}

    # Separate tokenized and non-tokenized fields
    tokenized_fields = [key for key in batch[0].keys() if any(
        key.startswith(f'{field}_') for field in ['title', 'ka', 'en']
    ) and key.endswith(('input_ids', 'attention_mask', 'token_type_ids'))]

    other_fields = [key for key in batch[0].keys() if key not in tokenized_fields]

    collated = {}

    # Handle tokenized fields (stack tensors)
    for field in tokenized_fields:
        if field in batch[0] and torch.is_tensor(batch[0][field]):
            collated[field] = torch.stack([item[field] for item in batch])

    # Handle other fields (collect in lists)
    for field in other_fields:
        collated[field] = [item[field] for item in batch]

    return collated


# Comprehensive testing
if __name__ == "__main__":
    print("=== Testing MultilingualDataset Class ===\n")

    # Create more comprehensive test data
    sample_data = pd.DataFrame({
        'title': ['Georgian News Article', 'English Literature', 'Mixed Content', 'Tech Article', 'Cultural Text',
                  None],
        'ka': ['ქართული სიახლეები დღეს', 'ქართული ლიტერატურა', 'შერეული კონტენტი', 'ტექნოლოგიები', 'კულტურული ტექსტი',
               ''],
        'en': ['Georgian news today', 'English literature piece', 'Mixed content here', 'Technology advances',
               'Cultural text content', 'Empty Georgian text'],
        'domain': ['news', 'literature', 'general', 'technology', 'culture', 'news'],
        'id': ['news_dataset_2024_01', 'lit_collection_classic', 'general_mixed_content', 'tech_articles_ai',
               'culture_heritage_2024', 'news_dataset_2024_02']
    })

    print("Original DataFrame:")
    print(sample_data)
    print(f"\nDataFrame shape: {sample_data.shape}")
    print("\n" + "=" * 50 + "\n")

    # Test 1: Basic dataset creation
    print("TEST 1: Basic Dataset Creation")
    try:
        dataset = MultilingualDataset(sample_data, include_metadata=True)
        print(f"✓ Dataset created successfully with {len(dataset)} items")
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
    print()

    # Test 2: Test __getitem__ method
    print("TEST 2: Getting Individual Items")
    try:
        first_item = dataset[0]
        print(f"✓ First item retrieved successfully")
        print(f"  - Title: {first_item['title']}")
        print(f"  - Domain: {first_item['domain']}")
        print(f"  - Dataset name: {first_item['dataset_name']}")
        print(f"  - Metadata: {first_item['metadata']}")
        print(f"  - Georgian text: {first_item['ka_text'][:30]}...")
        print(f"  - English text: {first_item['en_text'][:30]}...")
    except Exception as e:
        print(f"✗ Error getting item: {e}")
    print()

    # Test 3: Test get_domains method
    print("TEST 3: Getting Unique Domains")
    try:
        domains = dataset.get_domains()
        print(f"✓ Found {len(domains)} unique domains: {domains}")
    except Exception as e:
        print(f"✗ Error getting domains: {e}")
    print()

    # Test 4: Test get_dataset_names method
    print("TEST 4: Getting Unique Dataset Names")
    try:
        dataset_names = dataset.get_dataset_names()
        print(f"✓ Found {len(dataset_names)} unique dataset names: {dataset_names}")
    except Exception as e:
        print(f"✗ Error getting dataset names: {e}")
    print()

    # Test 5: Test get_languages method
    print("TEST 5: Getting Available Languages")
    try:
        languages = dataset.get_languages()
        print(f"✓ Available languages: {languages}")
    except Exception as e:
        print(f"✗ Error getting languages: {e}")
    print()

    # Test 6: Test get_by_domain method
    print("TEST 6: Filtering by Domain")
    try:
        news_items = dataset.get_by_domain('news')
        print(f"✓ Found {len(news_items)} news items")
        for i, item in enumerate(news_items):
            print(f"  News item {i + 1}: {item['title']}")

        tech_items = dataset.get_by_domain('technology')
        print(f"✓ Found {len(tech_items)} technology items")
        for i, item in enumerate(tech_items):
            print(f"  Tech item {i + 1}: {item['title']}")
    except Exception as e:
        print(f"✗ Error filtering by domain: {e}")
    print()

    # Test 7: Test get_by_dataset_name method
    print("TEST 7: Filtering by Dataset Name")
    try:
        news_dataset_items = dataset.get_by_dataset_name('news')
        print(f"✓ Found {len(news_dataset_items)} items from 'news' dataset")
        for i, item in enumerate(news_dataset_items):
            print(f"  Item {i + 1}: {item['title']} (ID: {item['id']})")
    except Exception as e:
        print(f"✗ Error filtering by dataset name: {e}")
    print()

    # Test 8: Test filter_by_domain method (creates new dataset)
    print("TEST 8: Creating Filtered Dataset by Domain")
    try:
        news_dataset = dataset.filter_by_domain('news')
        print(f"✓ Created filtered dataset with {len(news_dataset)} news items")
        print(f"  Original dataset size: {len(dataset)}")
        print(f"  Filtered dataset size: {len(news_dataset)}")

        # Test the filtered dataset
        first_filtered_item = news_dataset[0]
        print(f"  First filtered item domain: {first_filtered_item['domain']}")
    except Exception as e:
        print(f"✗ Error creating filtered dataset: {e}")
    print()

    # Test 9: Test sample method
    print("TEST 9: Random Sampling")
    try:
        sampled_dataset = dataset.sample(n=3, random_state=42)
        print(f"✓ Created sampled dataset with {len(sampled_dataset)} items")
        print("  Sampled items:")
        for i in range(len(sampled_dataset)):
            item = sampled_dataset[i]
            print(f"    {i + 1}. {item['title']} (Domain: {item['domain']})")
    except Exception as e:
        print(f"✗ Error creating sampled dataset: {e}")
    print()

    # Test 10: Test with tokenizer (mock tokenizer for testing)
    print("TEST 10: Testing with Mock Tokenizer")
    try:
        class MockTokenizer:
            def __call__(self, text, max_length=512, padding='max_length', truncation=True, return_tensors="pt"):
                # Simple mock tokenization
                tokens = text.split()[:max_length]
                input_ids = torch.tensor([hash(token) % 1000 for token in tokens] + [0] * (max_length - len(tokens)))
                attention_mask = torch.tensor([1] * len(tokens) + [0] * (max_length - len(tokens)))
                return {
                    'input_ids': input_ids.unsqueeze(0),
                    'attention_mask': attention_mask.unsqueeze(0)
                }


        mock_tokenizer = MockTokenizer()
        tokenized_dataset = MultilingualDataset(sample_data, tokenizer=mock_tokenizer, max_length=10)

        first_tokenized = tokenized_dataset[0]
        print(f"✓ Tokenized dataset created successfully")
        print(
            f"  Available tokenized fields: {[k for k in first_tokenized.keys() if 'input_ids' in k or 'attention_mask' in k]}")
        print(f"  Title input_ids shape: {first_tokenized['title_input_ids'].shape}")
        print(f"  Georgian input_ids shape: {first_tokenized['ka_input_ids'].shape}")
        print(f"  English input_ids shape: {first_tokenized['en_input_ids'].shape}")
    except Exception as e:
        print(f"✗ Error testing with tokenizer: {e}")
    print()

    # Test 11: Test collate function
    print("TEST 11: Testing Collate Function")
    try:
        # Test with tokenized data
        batch_items = [tokenized_dataset[i] for i in range(3)]
        collated_batch = collate_multilingual_batch(batch_items)

        print(f"✓ Collated batch successfully")
        print(f"  Batch keys: {list(collated_batch.keys())}")
        print(
            f"  Batch size for tokenized fields: {collated_batch['title_input_ids'].shape[0] if 'title_input_ids' in collated_batch else 'N/A'}")
        print(f"  Batch size for list fields: {len(collated_batch['title']) if 'title' in collated_batch else 'N/A'}")
    except Exception as e:
        print(f"✗ Error testing collate function: {e}")
    print()

    # Test 12: Test DataLoader creation
    print("TEST 12: Testing DataLoader Creation")
    try:
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)
        print(f"✓ DataLoader created successfully")

        # Test getting a batch
        first_batch = next(iter(dataloader))
        print(f"  First batch keys: {list(first_batch.keys())}")
        print(f"  Batch size: {len(first_batch['title'])}")
    except Exception as e:
        print(f"✗ Error creating DataLoader: {e}")
    print()

    # Test 13: Test error handling
    print("TEST 13: Testing Error Handling")
    try:
        # Test with missing columns
        bad_data = pd.DataFrame({'title': ['test'], 'missing_cols': ['test']})
        try:
            bad_dataset = MultilingualDataset(bad_data)
            print("✗ Should have raised an error for missing columns")
        except ValueError as e:
            print(f"✓ Correctly caught missing columns error: {e}")

        # Test accessing invalid index
        try:
            invalid_item = dataset[100]
            print("✗ Should have raised an error for invalid index")
        except (IndexError, KeyError) as e:
            print(f"✓ Correctly caught invalid index error: {type(e).__name__}")

        # Test getting dataset names without metadata
        dataset_no_metadata = MultilingualDataset(sample_data, include_metadata=False)
        try:
            names = dataset_no_metadata.get_dataset_names()
            print("✗ Should have raised an error when metadata is disabled")
        except ValueError as e:
            print(f"✓ Correctly caught metadata disabled error: {e}")

    except Exception as e:
        print(f"✗ Unexpected error in error handling test: {e}")
    print()

    print("=== All Tests Completed ===")
    print(f"Dataset successfully handles {len(dataset)} items across {len(dataset.get_domains())} domains")
    print(f"Unique dataset names found: {len(dataset.get_dataset_names())}")
    print(f"Languages supported: {', '.join(dataset.get_languages())}")