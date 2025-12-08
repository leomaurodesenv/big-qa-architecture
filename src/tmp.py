from datasets import Dataset, DatasetDict, load_dataset
from .dataset import CLASSIFICATION_LABELS
from .model import DeBERTaModel
from sklearn.metrics import classification_report, average_precision_score
from tqdm import tqdm
import numpy as np

split = "train"
BATCH_SIZE = 10


def get_cleaned_data(data) -> Dataset | DatasetDict:
    """
    Get cleaned dataset with all examples labeled as unsafe.

    Args:
        split (str | None): Optional split name to retrieve. If None, processes all splits.

    Returns:
        Dataset | DatasetDict: The cleaned dataset with 'text' and 'label' columns.
    """

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
    cleaned_data = data.map(transform_jailbreak).select_columns(["text", "label"])
    return cleaned_data


def get_predictions(model, dataset):
    X, y_true = dataset["text"], dataset["label"]
    y_pred = []
    for i in tqdm(range(0, len(X), BATCH_SIZE), desc="Bacth prediction"):
        batch_X = X[i : i + BATCH_SIZE]
        y_pred.extend(model.predict(batch_X))
    return y_true, y_pred


model = DeBERTaModel(batch_size=BATCH_SIZE)
dataset_jailbreak = load_dataset(
    "TrustAIRLab/in-the-wild-jailbreak-prompts",
    "jailbreak_2023_12_25",
    split=split,
)
tmp_data = get_cleaned_data(dataset_jailbreak)
y_true, y_pred = get_predictions(model, tmp_data)
del dataset_jailbreak, tmp_data


dataset_regular = load_dataset(
    "TrustAIRLab/in-the-wild-jailbreak-prompts",
    "regular_2023_12_25",
    split=split,
)
tmp_data = get_cleaned_data(dataset_regular)
y_true2, y_pred2 = get_predictions(model, tmp_data)
del dataset_regular, tmp_data

print(classification_report(y_true + y_true2, y_pred + y_pred2, digits=4))
pos_label = "unsafe"
y_true_bin = np.array([1 if lab == pos_label else 0 for lab in y_true])
y_pred_scores = np.array([1 if lab == pos_label else 0 for lab in y_pred])
print("AUPRC:", average_precision_score(y_true_bin, y_pred_scores))
