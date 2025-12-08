from abc import ABC, abstractmethod
from typing import Any
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

from src.model import CLASSIFICATION_LABELS


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders that provides common methods for accessing datasets.

    This class defines the interface and common functionality for loading and accessing
    Hugging Face datasets. Subclasses should implement the initialization logic to load
    their specific dataset.

    Example:
        >>> class MyDatasetLoader(BaseDatasetLoader):
        ...     def __init__(self, split=None):
        ...         self.split = split
        ...         self.dataset = load_dataset("my-dataset", split=split)
        >>> loader = MyDatasetLoader()
        >>> data = loader.get_data("train")
    """

    def get_split(self, split_name: str) -> Dataset:
        """
        Retrieve a specific split of the dataset.

        Args:
            split_name (str): The name of the dataset split to retrieve (e.g., "train", "test", "validation").

        Returns:
            Dataset: The requested dataset split.

        Raises:
            ValueError: If the split is not found in the dataset.
        """
        if isinstance(self.dataset, DatasetDict):
            if split_name in self.dataset:
                return self.dataset[split_name]
            else:
                available_splits = list(self.dataset.keys())
                raise ValueError(
                    f"Split '{split_name}' not found in the dataset. "
                    f"Available splits: {available_splits}"
                )
        else:
            # If dataset is already a single split, return it if the name matches
            if self.split == split_name:
                return self.dataset
            else:
                raise ValueError(
                    f"Dataset was loaded with split '{self.split}', "
                    f"but requested split is '{split_name}'."
                )

    def get_all_data(self) -> Dataset | DatasetDict:
        """
        Retrieve all available data from the dataset.

        Returns:
            Dataset | DatasetDict: All dataset splits as a DatasetDict if multiple splits exist,
                                  or a single Dataset if only one split was loaded.
        """
        return self.dataset

    def get_data(self, split: str | None = None) -> Dataset | DatasetDict:
        """
        Get dataset data, optionally filtered by split.

        Args:
            split (str | None): Optional split name to retrieve. If None, returns all data.

        Returns:
            Dataset | DatasetDict: The requested dataset split or all data.
        """
        if split:
            return self.get_split(split)
        return self.get_all_data()

    @abstractmethod
    def get_cleaned_data(self, split: str | None = None) -> Any:
        """
        Get cleaned dataset data with transformed columns and labels.
        """
        raise NotImplementedError("Subclasses must implement this method")


class DisasterTweetJailbreakingDataset(BaseDatasetLoader):
    """
    A class to load and access the Hugging Face dataset "IDA-SERICS/Disaster-tweet-jailbreaking".

    This dataset contains disaster-related tweets with jailbreaking prompts and their outputs.
    The dataset includes features such as 'prompt_attack', 'output_vittima', and 'label'.

    Example:
        >>> dataset_loader = DisasterTweetJailbreakingDataset()
        >>> train_data = dataset_loader.get_split("train")
        >>> all_data = dataset_loader.get_all_data()
    """

    def __init__(self, split: str | None = None):
        """
        Initialize the dataset loader.

        Args:
            split (str | None): Optional specific split to load. If None, loads all splits.
                                Common splits: "train", "test", "validation".
        """
        self.split = split
        if split:
            self.dataset = load_dataset(
                "IDA-SERICS/Disaster-tweet-jailbreaking", split=split
            )
        else:
            self.dataset = load_dataset("IDA-SERICS/Disaster-tweet-jailbreaking")

    def get_cleaned_data(self, split: str | None = None) -> Dataset | DatasetDict:
        """
        Get cleaned dataset data with transformed columns and labels.

        Transforms the dataset by:
        - Renaming 'prompt_attack' column to 'text'
        - Mapping 'label' values to CLASSIFICATION_LABELS:
          - 'jailbreak' → 'unsafe' (CLASSIFICATION_LABELS[0])
          - 'no-jailbreak' → 'safe' (CLASSIFICATION_LABELS[1])

        Args:
            split (str | None): Optional split name to retrieve. If None, processes all splits.

        Returns:
            Dataset | DatasetDict: The cleaned dataset with 'text' and 'label' columns.
        """
        # Get the dataset split(s)
        if split:
            data = self.get_split(split)
        else:
            data = self.get_all_data()

        def transform_labels(example):
            """Transform label values to CLASSIFICATION_LABELS."""
            label_mapping = {
                "jailbreak": CLASSIFICATION_LABELS[0],  # "unsafe"
                "no-jailbreak": CLASSIFICATION_LABELS[1],  # "safe"
            }
            return {
                "text": example["prompt_attack"],
                "label": label_mapping.get(example["label"], example["label"]),
            }

        # Apply transformation
        if isinstance(data, DatasetDict):
            cleaned_data = DatasetDict(
                {
                    split_name: split_data.map(transform_labels).select_columns(
                        ["text", "label"]
                    )
                    for split_name, split_data in data.items()
                }
            )
        else:
            # Single Dataset
            cleaned_data = data.map(transform_labels).select_columns(["text", "label"])

        return cleaned_data


class AegisDataset(BaseDatasetLoader):
    """
    Loader for the NVIDIA Aegis AI Content Safety Dataset.

    This loader attempts to be robust to the dataset's column names by
    auto-detecting a reasonable `text` column and a `label` column, then
    normalizing labels to the project's `CLASSIFICATION_LABELS` ("unsafe", "safe").

    Usage:
        loader = AegisDataset()
        train = loader.get_cleaned_data("train")
    """

    def __init__(self, split: str | None = None):
        self.split = split
        if split:
            self.dataset = load_dataset(
                "nvidia/Aegis-AI-Content-Safety-Dataset-1.0", split=split
            )
        else:
            self.dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0")

    def get_cleaned_data(self, split: str | None = None) -> Dataset | DatasetDict:
        if split:
            data = self.get_split(split)
        else:
            data = self.get_all_data()

        # The column name for text is explicitly "text"
        label_cols = ["labels_0", "labels_1", "labels_2", "labels_3", "labels_4"]

        def transform(example):
            # Build text field
            text_val = example["text"] if "text" in example else ""

            # Count how many of labels_0..labels_3 are 'Safe' (case-insensitive)
            safe_count = 0
            for col in label_cols:
                if col in example:
                    val = str(example[col]).strip().lower()
                    if val == "safe":
                        safe_count += 1

            mapped_label = (
                CLASSIFICATION_LABELS[1]
                if safe_count >= 2
                else CLASSIFICATION_LABELS[0]
            )

            return {"text": text_val, "label": mapped_label}

        # Apply transformation across splits or single dataset
        if isinstance(data, DatasetDict):
            cleaned_data = DatasetDict(
                {
                    split_name: split_data.map(transform).select_columns(
                        ["text", "label"]
                    )
                    for split_name, split_data in data.items()
                }
            )
        else:
            cleaned_data = data.map(transform).select_columns(["text", "label"])

        return cleaned_data


class TrustAIRLabJailbreakDataset(BaseDatasetLoader):
    """
    Loader for TrustAIRLab in-the-wild jailbreak prompts dataset.

    Loads the "jailbreak_2023_05_07" configuration which contains jailbreak prompts.
    All examples are labeled as CLASSIFICATION_LABELS[0] ("unsafe").

    Usage:
        loader = TrustAIRLabJailbreakDataset()
        data = loader.get_cleaned_data()
    """

    def __init__(self, split: str | None = None):
        """Load TrustAIRLab jailbreak dataset."""
        self.split = split
        if split:
            self.dataset_jailbreak = load_dataset(
                "TrustAIRLab/in-the-wild-jailbreak-prompts",
                "jailbreak_2023_12_25",
                split=split,
            )
            self.dataset_regular = load_dataset(
                "TrustAIRLab/in-the-wild-jailbreak-prompts",
                "regular_2023_12_25",
                split=split,
            )
        else:
            self.dataset_jailbreak = load_dataset(
                "TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25"
            )
            self.dataset_regular = load_dataset(
                "TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25"
            )

    def get_cleaned_data(self, split: str | None = None) -> Dataset | DatasetDict:
        """
        Get cleaned dataset with all examples labeled as unsafe.

        Args:
            split (str | None): Optional split name to retrieve. If None, processes all splits.

        Returns:
            Dataset | DatasetDict: The cleaned dataset with 'text' and 'label' columns.
        """
        if split:
            data_jailbreak = self.dataset_jailbreak[split]
            data_regular = self.dataset_regular[split]
        else:
            data_jailbreak = self.dataset_jailbreak
            data_regular = self.dataset_regular

        def transform_jailbreak(example):
            """Transform jailbreak data to text and label."""
            return {
                "text": example.get("prompt", ""),
                "label": CLASSIFICATION_LABELS[0],  # unsafe
            }

        def transform_regular(example):
            """Transform regular data to text and label."""
            return {
                "text": example.get("prompt", ""),
                "label": CLASSIFICATION_LABELS[1],  # safe
            }

        # Apply transformations
        jailbreak_cleaned = data_jailbreak.map(transform_jailbreak).select_columns(
            ["text", "label"]
        )
        regular_cleaned = data_regular.map(transform_regular).select_columns(
            ["text", "label"]
        )

        # Combine both datasets
        cleaned_data = concatenate_datasets([jailbreak_cleaned, regular_cleaned])

        return cleaned_data
