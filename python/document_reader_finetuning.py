"""
This script fine-tunes a Document Reader models for a question-answering task
on a sports-related dataset. It includes data preprocessing, model training, evaluation, and
inference testing. The script also supports saving and uploading the trained model and tokenizer
to Hugging Face.

Key Steps:
1. Load environment variables and authenticate with Hugging Face.
2. Load and preprocess the sports dataset for training and validation.
3. Fine-tune a pre-trained DistilBERT model for question-answering.
4. Evaluate the model using the SQuAD v2 metric.
5. Save and optionally upload the trained model and tokenizer to Hugging Face.
6. Perform inference testing with a sample question and context.

Environment Variables:
- `HUGGINGFACE_TOKEN`: Token for authenticating with Hugging Face.
- `HUGGINGFACE_PROFILE`: Hugging Face profile name, for saving the model.
"""

import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from utils import DocReader, Sports
from utils import (
    preprocessing_dataset,
    load_sports_dataset,
    encapsulate_metrics,
)
from transformers import (
    Trainer,
    pipeline,
    AutoTokenizer,
    TrainingArguments,
    DefaultDataCollator,
    EarlyStoppingCallback,
    AutoModelForQuestionAnswering,
)


# Read environment variables from .env file
load_dotenv()


# Parse command-line arguments to get DocReader and Sports
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a Document Reader for QA.")
    parser.add_argument(
        "--doc_reader",
        type=str,
        required=True,
        choices=[reader.name for reader in DocReader],
        help="Specify the document reader model (e.g., DistilBERT, BERT).",
    )
    parser.add_argument(
        "--sport",
        type=str,
        required=True,
        choices=[sport.name for sport in Sports],
        help="Specify the sport dataset (e.g., BASKETBALL, SOCCER).",
    )
    args = parser.parse_args()
    return DocReader[args.doc_reader], Sports[args.sport]


# Get the document reader and sport dataset from command-line arguments
# Generate a simplified file name based on the selected DOC_READER and SPORT
DOC_READER, SPORT = parse_arguments()
model_name = f"{DOC_READER.name.lower()}-{SPORT.name.lower()}-qa"
print("Running:", model_name)
print("Model name:", DOC_READER.value)

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
huggingface_profile = os.getenv("HUGGINGFACE_PROFILE")
login(token=huggingface_token)

# Load dataset
dataset_train = load_sports_dataset(sport=SPORT, split="train")
dataset_validation = load_sports_dataset(sport=SPORT, split="validation")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(DOC_READER.value)
model = AutoModelForQuestionAnswering.from_pretrained(DOC_READER.value)

# Preprocess training and validation datasets
cleaned_data_train, _, filtered_data_train = preprocessing_dataset(
    tokenizer=tokenizer, dataset=dataset_train
)
cleaned_data_validation, _, filtered_data_validation = preprocessing_dataset(
    tokenizer=tokenizer, dataset=dataset_validation
)

# Checking the size of preprocessed datasets
print(
    f"Original: {len(dataset_train)}, Cleaned: {len(cleaned_data_train)}, Filtered: {len(filtered_data_train)}"
)
print(
    f"Original: {len(dataset_validation)}, Cleaned: {len(cleaned_data_validation)}, Filtered: {len(filtered_data_validation)}"
)

# Training arguments
data_collator = DefaultDataCollator()
training_args = TrainingArguments(
    output_dir=model_name,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    logging_steps=50,
    num_train_epochs=100,
    weight_decay=0.01,
    push_to_hub=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=filtered_data_train,
    eval_dataset=filtered_data_validation,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=encapsulate_metrics(
        validation_dataset=filtered_data_validation, tokenizer=tokenizer
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

# Run fine-tuning
trainer.train()

# Save tokenizer locally
tokenizer.save_pretrained(model_name)
# Upload tokenizer to Hugging Face
repo_id = f"{huggingface_profile}/{model_name}"
api = HfApi()
api.upload_folder(
    folder_path=model_name,
    repo_id=repo_id,
    commit_message="feat: add tokenizer",
)

# Publish model to Hugging Face
trainer.push_to_hub()

# Inference testing
question = "Who owns the building?"
context = "(See Memphis Summer Storm of 2003.) It was built at a cost of US$250 million and is owned by the City of Memphis, naming rights were purchased by one of Memphis' most well-known businesses, FedEx, for $92 million. FedExForum was financed using $250 million of public bonds, which were issued by the Memphis Public Building Authority (PBA)."
question_answerer = pipeline(
    task="question-answering", model=repo_id, tokenizer=repo_id
)
answer = question_answerer(question=question, context=context)

print("Test Inference from HuggingFace")
print("Question:", question)
print("Context:", context)
print("Answer:", answer)
