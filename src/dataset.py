from abc import ABC
from datasets import Dataset, DatasetDict, load_dataset


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
