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
"""

import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login
from utils import DocReader, Sports
from utils import (
    preprocess_function,
    clean_dataset,
    filter_dataset,
    load_sports_dataset,
)
from transformers import (
    AutoTokenizer,
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
model_name = f"{DOC_READER.name.lower()}-{SPORT.name.lower()}"
print("Running:", model_name)
print("Model name:", DOC_READER.value)

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=huggingface_token)

# Load dataset
dataset_train = load_sports_dataset(sport=SPORT.value, split="train")
dataset_validation = load_sports_dataset(sport=SPORT.value, split="validation")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(DOC_READER.value)
model = AutoModelForQuestionAnswering.from_pretrained(DOC_READER.value)


# Preprocessing sports dataset
def preprocessing_dataset(tokenizer, dataset):
    cleaned_data = clean_dataset(dataset=dataset)
    preprocessed_data = cleaned_data.map(
        lambda x: preprocess_function(tokenizer=tokenizer, examples=x), batched=True
    )
    filtered_data = filter_dataset(dataset=preprocessed_data)
    return cleaned_data, preprocessed_data, filtered_data


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


# def compute_metrics(p):
#     metric = load("squad_v2")
#     predictions, labels = p
#     start_preds = predictions[0].argmax(axis=1)
#     end_preds = predictions[1].argmax(axis=1)

#     formatted_predictions = [
#         {
#             "id": str(i),
#             "prediction_text": tokenizer.decode(
#                 filtered_data_validation[i]["input_ids"][
#                     start_preds[i] : end_preds[i] + 1
#                 ],  # Corrigido
#                 skip_special_tokens=True,
#             ),
#             "no_answer_probability": 0.0,
#         }
#         for i in range(len(start_preds))
#     ]

#     references = [
#         {
#             "id": str(i),
#             "answers": {
#                 "text": [filtered_data_validation[i]["answer"]["text"]],
#                 "answer_start": [filtered_data_validation[i]["answer"]["offset"][0]],
#             },
#         }
#         for i in range(len(filtered_data_validation))
#     ]  # Corrigido

#     return metric.compute(predictions=formatted_predictions, references=references)


# data_collator = DefaultDataCollator()

# training_args = TrainingArguments(
#     output_dir="distilbert-qasports",
#     evaluation_strategy="steps",
#     eval_steps=500,
#     save_strategy="steps",
#     learning_rate=1e-5,
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=2,
#     per_device_eval_batch_size=16,
#     num_train_epochs=50,
#     weight_decay=0.01,
#     push_to_hub=False,  # True
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     logging_steps=100,
#     fp16=True,
#     report_to="none",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=filtered_data_train,  # TODO: Processed training dataset
#     eval_dataset=filtered_data_validation,  # TODO: Processed validation dataset
#     data_collator=data_collator,  # TODO: Correct data collator
#     # compute_metrics=compute_metrics, # TODO: Correct the computing metrics
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
# )

# trainer.train()


# # TODO: Upload the model to hugging face
# # Save tokenizer locally
# # tokenizer.save_pretrained(save_name)

# # Upload tokenizer to Hugging Face
# # api = HfApi()
# # api.upload_folder(
# #     folder_path=save_name,
# #     repo_id=f"laurafcamargos/{save_name}",
# #     commit_message="Adding tokenizer",
# # )

# # Publish model to Hugging Face
# # trainer.push_to_hub()

# # Inference testing
# question = "Who owns the building?"
# context = "(See Memphis Summer Storm of 2003.) It was built at a cost of US$250 million and is owned by the City of Memphis, naming rights were purchased by one of Memphis' most well-known businesses, FedEx, for $92 million. FedExForum was financed using $250 million of public bonds, which were issued by the Memphis Public Building Authority (PBA)."

# question_answerer = pipeline(
#     task="question-answering", model=model, tokenizer=tokenizer
# )
# answer = question_answerer(question=question, context=context)

# print(answer)
