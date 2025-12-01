import logging
import os
from typing import Any
from abc import ABC, abstractmethod

from transformers import AutoTokenizer, pipeline
from optimum.intel import OVModelForSequenceClassification


CLASSIFICATION_LABELS = ["unsafe", "safe"]

# Set up module-level logger
logger = logging.getLogger(__name__)


def _setup_logger(debug: bool) -> logging.Logger:
    """
    Set up and configure a logger based on debug mode.

    Args:
        debug (bool): Whether debug mode is enabled.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_level = logging.DEBUG if debug else logging.WARNING
    logger.setLevel(log_level)

    # Only add handler if one doesn't exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Update existing handler level and formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        for handler in logger.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


class BaseModel(ABC):
    """
    Abstract base class for model wrappers that provides common methods for model access and predictions.

    This class defines the interface and common functionality for loading models and making predictions.
    Subclasses should implement the initialization logic to load their specific model.

    Example:
        >>> class MyModel(BaseModel):
        ...     def __init__(self):
        ...         self.model = load_my_model()
        >>> model = MyModel()
        >>> prediction = model.predict("sample text")
    """

    @abstractmethod
    def get_model(self) -> Any:
        """
        Retrieve the underlying model object.

        Returns:
            Any: The model object (type depends on the implementation).
        """
        pass

    @abstractmethod
    def predict(self, texts: list[str], **kwargs) -> list[Any]:
        """
        Make predictions on a batch of text inputs.

        Args:
            texts (list[str]): A list of input texts to make predictions on.
            **kwargs: Additional keyword arguments for the predictions.

        Returns:
            list[Any]: A list of prediction results.
        """
        pass


class ArchGuardModel(BaseModel):
    """
    A model wrapper for the Arch-Guard text classification model using OpenVINO.

    This class uses OpenVINO (OVModelForSequenceClassification) to load and use the
    "katanemolabs/Arch-Guard-cpu" model for text classification tasks.

    Example:
        >>> model = ArchGuardModel()
        >>> prediction = model.predict(["This is a sample text"])
        >>> proba = model.predict_proba(["This is a sample text"])
    """

    def __init__(self, device: str = "cpu", debug: bool = False):
        """
        Initialize the Arch-Guard model using OpenVINO.

        Args:
            device (str): The device to run the model on. Default is "cpu".
            debug (bool | None): Enable debug mode for verbose output. If None, uses DEBUG env var.
        """
        model_name = f"katanemolabs/Arch-Guard-{device}"

        self.model = OVModelForSequenceClassification.from_pretrained(
            model_name, device_map=device, low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.device = device
        self.UNSAFE_TOKEN = "JAILBREAK"
        self.debug = debug
        self.logger = _setup_logger(self.debug)

    def get_model(self) -> Any:
        """
        Retrieve the underlying model object.

        Returns:
            Any: The OpenVINO model object.
        """
        return self.model

    def predict(self, texts: list[str], **kwargs) -> list[dict[str, Any]]:
        """
        Make text classification predictions on a batch of text inputs.

        Args:
            texts (list[str]): A list of input texts to classify.
            **kwargs: Additional keyword arguments (e.g., return_all_scores).

        Returns:
            list[dict[str, Any]]: A list of prediction dictionaries, one for each input text.
                                  Each dict contains 'label' and 'score' keys.
        """
        import torch

        # Tokenize inputs
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, **kwargs
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get predicted labels and scores
        predicted_ids = torch.argmax(probabilities, dim=-1)
        scores = torch.max(probabilities, dim=-1).values

        # Get label names from model config
        id2label = self.model.config.id2label

        results = []
        for pred_id, score in zip(predicted_ids, scores):
            label = id2label[pred_id.item()]
            results.append({"label": label, "score": score.item()})
            self.logger.debug("prediction: %s", results[-1])

        return [
            CLASSIFICATION_LABELS[0]
            if self.UNSAFE_TOKEN == str(result["label"])
            else CLASSIFICATION_LABELS[1]
            for result in results
        ]


class SamsungJailbreakFilterModel(BaseModel):
    """
    A model wrapper for the Samsung SGuard-JailbreakFilter-2B-v1 text generation model using transformers pipeline.

    This class uses the transformers pipeline to load and use the
    "SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1" model for text generation tasks.

    Example:
        >>> model = SamsungJailbreakFilterModel()
        >>> prediction = model.predict(["Who are you?"])
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the Samsung JailbreakFilter model using transformers pipeline.

        Args:
            debug (bool | None): Enable debug mode for verbose output. If None, uses DEBUG env var.
        """
        self.pipe = pipeline(
            "text-generation", model="SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1"
        )
        self.UNSAFE_TOKEN = "unsafe"
        self.max_new_tokens = 1
        self.debug = debug
        self.logger = _setup_logger(self.debug)

    def get_model(self) -> Any:
        """
        Retrieve the underlying pipeline object.

        Returns:
            Any: The transformers pipeline object.
        """
        return self.pipe

    def predict(self, texts: list[str], **kwargs) -> list[Any]:
        """
        Make text generation predictions on a batch of text inputs.

        Args:
            texts (list[str]): A list of input texts to generate responses for.
            **kwargs: Additional keyword arguments for the pipeline.

        Returns:
            list[Any]: A list of prediction results from the pipeline.
        """
        results = []
        for text in texts:
            messages = [
                {"role": "user", "content": text},
            ]
            result = self.pipe(
                messages, **kwargs | {"max_new_tokens": self.max_new_tokens}
            )
            results.append(result[0]["generated_text"][1]["content"])
            self.logger.debug(
                "prediction: %s", result[0]["generated_text"][1]["content"]
            )

        return [
            CLASSIFICATION_LABELS[0]
            if self.UNSAFE_TOKEN in str(result)
            else CLASSIFICATION_LABELS[1]
            for result in results
        ]


class LlamaGuardModel(SamsungJailbreakFilterModel):
    """
    A model wrapper for the Llama-Guard-3-8B text generation model using transformers pipeline.

    This class uses the transformers pipeline to load and use the
    "meta-llama/Llama-Guard-3-8B" model for text generation tasks.

    Example:
        >>> model = LlamaGuardModel()
        >>> prediction = model.predict(["Who are you?"])
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the Llama-Guard model using transformers pipeline.

        Args:
            debug (bool | None): Enable debug mode for verbose output. If None, uses DEBUG env var.
        """
        # meta-llama/Llama-Guard-3-8B
        # meta-llama/Llama-Guard-3-1B
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-Guard-3-1B")
        self.UNSAFE_TOKEN = "unsafe"
        self.max_new_tokens = 2
        self.debug = debug
        self.logger = _setup_logger(self.debug)
