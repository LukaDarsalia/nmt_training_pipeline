"""
S3 Data Loader Module

A utility module for uploading and downloading files/folders to/from AWS S3.
Provides a simple interface for S3 operations with progress tracking, plus
utilities for composing multiple loading operations.
"""

import datetime
import os
from pathlib import Path
from typing import Union, List, Callable

import boto3
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv('.env')


def get_s3_loader(bucket_name: str) -> 'S3DataLoader':
    """
    Create and return an S3DataLoader instance with credentials from environment variables.

    Args:
        bucket_name (str): Name of the S3 bucket to work with

    Returns:
        S3DataLoader: Configured S3DataLoader instance

    Raises:
        KeyError: If required environment variables are not set
    """
    s3_loader = S3DataLoader(
        bucket=bucket_name,
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    return s3_loader


def generate_folder_name() -> str:
    """
    Generate a timestamp-based folder name for organizing uploads.

    Returns:
        str: Folder name in format 'YYYY-MM-DD_HH-MM-SS'

    Example:
        >>> generate_folder_name()
        '2024-03-15_14-30-45'
    """
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    return folder_name


class S3DataLoader:
    """
    A utility class for uploading and downloading files/folders to/from AWS S3.

    This class provides a simple interface for S3 operations with automatic
    file/folder detection and progress tracking using tqdm.

    Attributes:
        access_key (str): AWS access key ID
        secret_key (str): AWS secret access key
        bucket (str): S3 bucket name
        s3_client: Boto3 S3 client instance
    """

    def __init__(self, bucket: str, access_key: str, secret_key: str):
        """
        Initialize the S3DataLoader with AWS credentials and bucket name.

        Args:
            bucket (str): Name of the S3 bucket
            access_key (str): AWS access key ID
            secret_key (str): AWS secret access key
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

    def download(self, path: Union[str, Path]) -> None:
        """
        Download a file or folder from S3 to local filesystem.

        Automatically detects whether the path refers to a file or folder
        based on the presence of a file extension.

        Args:
            path (Union[str, Path]): S3 path to download. If it contains a file
                                   extension, treated as a file; otherwise as a folder.

        Example:
            >>> loader.download('data/file.txt')  # Downloads single file
            >>> loader.download('data/folder')    # Downloads entire folder
        """
        path = Path(path)

        if "." in path.name:
            self._download_file(path)
        else:
            self._download_folder(str(path))

    def _download_file(self, filepath: Path) -> None:
        """
        Download a single file from S3 to local filesystem.

        Creates parent directories if they don't exist.

        Args:
            filepath (Path): Path to the file in S3 and local destination
        """
        # Create parent directories if they don't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        self.s3_client.download_file(
            self.bucket,
            str(filepath),
            str(filepath)
        )

    def _download_folder(self, remote_dir_name: str) -> None:
        """
        Download an entire folder from S3 to local filesystem.

        Recursively downloads all files within the specified folder,
        maintaining the directory structure.

        Args:
            remote_dir_name (str): Name/path of the folder in S3 to download
        """
        # Create S3 resource for folder operations
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(self.bucket)

        # Get all objects with the specified prefix
        all_objects = list(bucket.objects.filter(Prefix=remote_dir_name))

        # Download each object with progress bar
        for obj in tqdm(all_objects, desc=f"Downloading folder {remote_dir_name}"):
            # Create local directory structure if it doesn't exist
            local_dir = os.path.dirname(obj.key)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Skip directory markers (keys ending with '/')
            if obj.key.endswith('/'):
                continue

            # Download the file to the same relative path locally
            bucket.download_file(obj.key, obj.key)

    def upload(self, path: Union[str, Path]) -> None:
        """
        Upload a file or folder from local filesystem to S3.

        Automatically detects whether the path is a file or directory
        and calls the appropriate upload method.

        Args:
            path (Union[str, Path]): Local path to upload. Can be a file or directory.

        Example:
            >>> loader.upload('local_file.txt')     # Uploads single file
            >>> loader.upload('local_folder')       # Uploads entire folder
        """
        path = Path(path)

        if path.is_dir():
            self._upload_folder(str(path))
        else:
            self._upload_file(path)

    def _upload_file(self, filepath: Union[str, Path]) -> None:
        """
        Upload a single file from local filesystem to S3.

        Args:
            filepath (Union[str, Path]): Path to the local file to upload
        """
        self.s3_client.upload_file(
            str(filepath),
            self.bucket,
            str(filepath)
        )

    def _upload_folder(self, folder_path: str) -> None:
        """
        Upload an entire folder from local filesystem to S3.

        Recursively uploads all files within the specified folder,
        maintaining the directory structure in S3.

        Args:
            folder_path (str): Path to the local folder to upload
        """
        # Get all files in the folder recursively
        all_files = list(Path(folder_path).rglob("*"))

        # Upload each file with progress bar
        for file_path in tqdm(all_files, desc=f"Uploading folder {folder_path}"):
            # Only upload files, not directories
            if not file_path.is_dir():
                self._upload_file(file_path)


class ComposeLoading:
    """
    A utility class for composing and executing multiple loading functions sequentially.

    This class allows you to group multiple loading functions together and execute
    them in sequence with progress feedback. Useful for orchestrating complex
    loading workflows with multiple data sources or processing steps.

    Attributes:
        load_functions (List[Callable]): List of callable functions to execute
    """

    def __init__(self, load_functions: List[Callable]) -> None:
        """
        Initialize the ComposeLoading with a list of functions to execute.

        Args:
            load_functions (List[Callable]): List of callable functions that will be executed
                                           sequentially. Can include regular functions or
                                           functools.partial objects.

        Example:
            >>> def load_data(): pass
            >>> def load_models(): pass
            >>> composer = ComposeLoading([load_data, load_models])
        """
        self.load_functions = load_functions

    def __call__(self) -> None:
        """
        Execute all loading functions in sequence with progress feedback.

        Calls each function in the order they were provided and prints a
        success message after each completion. Handles both regular functions
        and functools.partial objects gracefully.

        Example:
            >>> composer = ComposeLoading([func1, func2])
            >>> composer()  # Executes func1, then func2
            Loaded func1!
            Loaded func2!
        """
        for function in self.load_functions:
            # Execute the function
            function()

            # Print success message with function name
            try:
                function_name = function.__name__
                print(f"Loaded {function_name}!")
            except AttributeError:
                # Handle functools.partial objects which don't have __name__
                function_name = function.func.__name__
                print(f"Loaded {function_name}!")

    def __repr__(self) -> str:
        """
        Return a formatted string representation of the ComposeLoading object.

        Returns:
            str: Multi-line string showing the class name and all contained functions

        Example:
            >>> print(composer)
            ComposeLoading(
                <function load_data at 0x...>
                <function load_models at 0x...>
            )
        """
        format_string = f"{self.__class__.__name__}("

        for function in self.load_functions:
            format_string += f"\n    {function}"

        format_string += "\n)"
        return format_string
