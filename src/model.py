from typing import Any
from abc import ABC, abstractmethod

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from optimum.intel import OVModelForSequenceClassification


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

    def __init__(self, device: str = "cpu"):
        """
        Initialize the Arch-Guard model using OpenVINO.

        Args:
            device (str): The device to run the model on. Default is "cpu".
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
        for i, (pred_id, score) in enumerate(zip(predicted_ids, scores)):
            label = id2label[pred_id.item()]
            results.append({"label": label, "score": score.item()})

        return results


class SamsungJailbreakFilterModel(BaseModel):
    """
    A model wrapper for the Samsung SGuard-JailbreakFilter-2B-v1 text generation model using transformers pipeline.

    This class uses the transformers pipeline to load and use the
    "SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1" model for text generation tasks.

    Example:
        >>> model = SamsungJailbreakFilterModel()
        >>> prediction = model.predict(["Who are you?"])
    """

    def __init__(self):
        """
        Initialize the Samsung JailbreakFilter model using transformers pipeline.
        """
        self.pipe = pipeline(
            "text-generation", model="SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1"
        )
        self.UNSAFE_TOKEN = "unsafe"
        self.max_new_tokens = 1

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
            print("test:", result[0]["generated_text"][1]["content"])

        return [
            "unsafe" if self.UNSAFE_TOKEN in str(result) else "safe"
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

    def __init__(self):
        """
        Initialize the Llama-Guard model using transformers pipeline.
        """
        # meta-llama/Llama-Guard-3-8B
        # meta-llama/Llama-Guard-3-1B
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-Guard-3-1B")
        self.UNSAFE_TOKEN = "unsafe"
        self.max_new_tokens = 2
