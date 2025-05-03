import ast
import evaluate
from enum import Enum
from datasets import Dataset, load_dataset


class DocReader(str, Enum):
    """
    DocReader is a utility class that provides predefined options for various
    pre-trained transformer models used in question-answering tasks. Each model
    is represented as a class attribute with its corresponding model identifier.
    """

    BERT = "deepset/bert-base-uncased-squad2"
    RoBERTa = "deepset/roberta-base-squad2"
    MiniLM = "deepset/minilm-uncased-squad2"
    DistilBERT = "distilbert-base-uncased-distilled-squad"
    ELECTRA = "deepset/electra-base-squad2"


class Sports(str, Enum):
    """
    A class representing different types of subset on sports dataset.
    Each sport is represented as a class attribute with its corresponding name.
    """

    BASKETBALL = "basketball"
    FOOTBALL = "football"
    SOCCER = "soccer"
    ALL = ""


def clean_dataset(dataset: Dataset) -> Dataset:
    """
    Cleans and validates a dataset by processing each example to ensure it meets specific criteria.

    Args:
        data (Dataset): The input dataset containing examples with 'question', 'context', and 'answer' fields.

    Returns:
        Dataset: A new dataset containing only the cleaned and validated examples.

    Each example in the input dataset is expected to have the following structure:
        - 'question' (str): The question text.
        - 'context' (str): The context text.
        - 'answer' (str or dict): The answer, which can be a string (to be converted to a dictionary) or a dictionary.
    """
    cleaned_data = []
    for example in dataset:
        try:
            # Convert 'answer' from string to dictionary
            answer = (
                ast.literal_eval(example["answer"])
                if isinstance(example["answer"], str)
                else example["answer"]
            )
            context = example["context"]

            # Validation
            if (
                isinstance(answer, dict)
                and "text" in answer
                and "offset" in answer
                and isinstance(answer["offset"], list)
                and len(answer["offset"]) == 2
                and answer["offset"][1] > answer["offset"][0]
                and answer["offset"][1]
                <= len(context)  # Checks if the offset is within the context
            ):
                cleaned_data.append(
                    {
                        "question": example["question"],
                        "context": context,
                        "answer": answer,
                    }
                )
        except (SyntaxError, ValueError, KeyError):
            continue

    return Dataset.from_list(cleaned_data)


def preprocess_function(tokenizer, examples):
    """
    Preprocesses input examples for a question-answering task by tokenizing the questions
    and contexts, and computing the start and end positions of the answers within the tokenized context.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the input text.
        examples (dict): A dictionary containing the following keys:
            - "question" (list of str): The list of questions.
            - "context" (list of str): The list of contexts corresponding to the questions.
            - "answer" (list of dict): A list of dictionaries, each containing:
                - "text" (str): The answer text.
                - "offset" (tuple of int): The character start and end offsets of the answer in the context.

    Returns:
        dict: A dictionary containing the tokenized inputs with the following keys:
            - All keys returned by the tokenizer, such as "input_ids", "attention_mask", etc.
            - "start_positions" (list of int): The start token indices of the answers in the tokenized context.
            - "end_positions" (list of int): The end token indices of the answers in the tokenized context.
    """
    # Tokenize the questions and contexts
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",  # Truncate only the context
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answer"][i]

        # Fallback for invalid answers
        if not answer.get("text"):
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["offset"][0]
        end_char = answer["offset"][1]
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context in the tokenized text
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1

        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        # If the context was truncated, adjust the boundaries
        tokenized_context_start = (
            offsets[context_start][0] if context_start < len(offsets) else 0
        )
        tokenized_context_end = offsets[context_end][1] if context_end >= 0 else 0

        # Check if the answer is within the tokenized context
        if start_char > tokenized_context_end or end_char < tokenized_context_start:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Adjust the offsets to fit within the truncated context
            start_char = max(start_char, tokenized_context_start)
            end_char = min(end_char, tokenized_context_end)

            # Find tokens
            start_idx = context_start
            while start_idx <= context_end and offsets[start_idx][0] <= start_char:
                start_idx += 1
            start_pos = start_idx - 1

            end_idx = context_end
            while end_idx >= context_start and offsets[end_idx][1] >= end_char:
                end_idx -= 1
            end_pos = end_idx + 1

            start_positions.append(start_pos)
            end_positions.append(end_pos)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def filter_dataset(dataset: Dataset) -> Dataset:
    """
    Filters a dataset to remove invalid samples based on specific conditions.

    This function filters out samples where:
    - The "start_positions" field is 0.
    - The "end_positions" field is 0.
    - The "start_positions" value is greater than the "end_positions" value.

    Args:
        dataset (Dataset): The input dataset to be filtered. Each sample in the dataset
                           is expected to have "start_positions" and "end_positions" fields.

    Returns:
        Dataset: A filtered dataset containing only valid samples.
    """
    return dataset.filter(
        lambda x: x["start_positions"]
        != 0  # drop samples where {'text': '', 'offset': [0, 0]}
        and x["end_positions"] != 0
        and x["start_positions"] <= x["end_positions"]
    )


def preprocessing_dataset(tokenizer, dataset):
    """
    Preprocesses a dataset for a question-answering task by cleaning, tokenizing, and filtering the dataset.
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the input text.
        dataset (Dataset): The input dataset containing examples with 'question', 'context', and 'answer' fields.
    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing three datasets:
            - cleaned_data: The cleaned dataset.
            - preprocessed_data: The preprocessed dataset with tokenized inputs.
            - filtered_data: The filtered dataset with valid samples.
    """
    cleaned_data = clean_dataset(dataset=dataset)
    preprocessed_data = cleaned_data.map(
        lambda x: preprocess_function(tokenizer=tokenizer, examples=x), batched=True
    )
    filtered_data = filter_dataset(dataset=preprocessed_data)
    return cleaned_data, preprocessed_data, filtered_data


def load_sports_dataset(sport: Sports, split: str):
    """
    Loads the QASports dataset for a specific sport and split.
    Args:
        sport (Sports): The sport to load the dataset for. Options are 'basketball', 'football', 'soccer'.
        split (str): The split of the dataset to load. Options are 'train', 'validation', 'test'.
    Returns:
        Dataset: The loaded dataset.
    """
    if sport not in Sports:
        raise ValueError(f"Invalid sport '{sport}'.")
    return load_dataset("PedroCJardim/QASports", sport.value, split=split)


def encapsulate_metrics(validation_dataset, tokenizer):
    """
    Encapsulates the evaluation metrics for the SQuAD v2 dataset.
    This function uses the `evaluate` library to compute the F1 and Exact Match (EM) scores.
    The function takes the validation dataset and tokenizer as input.
    """

    def compute_metrics(eval_preds):
        """
        Computes evaluation metrics for the SQuAD v2 dataset.
        This function uses the `evaluate` library to compute the F1 and Exact Match (EM) scores.
        The function takes the validation dataset, tokenizer, and evaluation predictions as input.
        It formats the predictions and references according to the expected structure for SQuAD v2.
        Args:
            validation_dataset (Dataset): The validation dataset containing the ground truth answers.
            tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the input text.
            eval_preds (tuple): A tuple containing the model predictions and labels.
        Returns:
            dict: A dictionary containing the computed F1 and EM scores.
        Example:
            EvaluationModule(
                name: "squad_v2",
                module_type: "metric",
                features: {
                    'predictions': {
                        'id': Value(dtype='string', id=None),
                        'prediction_text': Value(dtype='string', id=None),
                        'no_answer_probability': Value(dtype='float32', id=None)},
                        'references': {'id': Value(dtype='string', id=None),
                        'answers': Sequence(feature={'text': Value(dtype='string', id=None),
                        'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None)
                    }
                },
        """
        metric = evaluate.load("shalakasatheesh/squad_v2")
        predictions, _ = eval_preds.predictions, eval_preds.label_ids
        start_preds = predictions[0].argmax(axis=1)
        end_preds = predictions[1].argmax(axis=1)

        formatted_predictions = [
            {
                "id": str(i),
                "prediction_text": tokenizer.decode(
                    validation_dataset[i]["input_ids"][
                        start_preds[i] : end_preds[i] + 1
                    ],
                    skip_special_tokens=True,
                ),
                "no_answer_probability": 0.0,
            }
            for i in range(len(start_preds))
        ]

        references = [
            {
                "id": str(i),
                "answers": {
                    "text": [validation_dataset[i]["answer"]["text"]],
                    "answer_start": [validation_dataset[i]["answer"]["offset"][0]],
                },
            }
            for i in range(len(validation_dataset))
        ]
        return metric.compute(predictions=formatted_predictions, references=references)

    # Return the encapsulated compute_metrics function
    return compute_metrics
