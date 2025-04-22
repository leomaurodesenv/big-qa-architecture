import os
from utils import DocReader, Sports
from utils import (
    preprocess_function,
    load_and_clean_data,
    filter_dataset,
    load_sports_dataset,
)
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    pipeline,
)
from huggingface_hub import login
from datasets import load
from dotenv import load_dotenv

# Read environment variables from .env file
load_dotenv()

# Specify the document reader and Sport Dataset
DOC_READER = DocReader.DistilBERT
SPORT = Sports.BASKETBALL
print(SPORT, DOC_READER)

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=huggingface_token)

# Load dataset
dataset_train = load_sports_dataset(sport=SPORT, split="train[0:100]")
dataset_validation = load_sports_dataset(sport=SPORT, split="validation[0:10]")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(DOC_READER)
model = AutoModelForQuestionAnswering.from_pretrained(DOC_READER)

# Complete preprocessing pipeline
cleaned_data_train = load_and_clean_data(dataset_train)
cleaned_data_validation = load_and_clean_data(dataset_validation)
processed_data_train = cleaned_data_train.map(
    lambda x: preprocess_function(tokenizer=tokenizer, examples=x), batched=True
)
processed_data_validation = cleaned_data_validation.map(
    lambda x: preprocess_function(tokenizer=tokenizer, examples=x), batched=True
)
filtered_data_train = filter_dataset(processed_data_train)
filtered_data_validation = filter_dataset(processed_data_validation)

# Checking the size of preprocessed vectors
print(
    f"Original: {len(dataset_train)}, Cleaned: {len(cleaned_data_train)}, Filtered: {len(filtered_data_train)}"
)
print(
    f"Original: {len(dataset_validation)}, Cleaned: {len(cleaned_data_validation)}, Filtered: {len(filtered_data_validation)}"
)

# Checking a sample vector
print(filtered_data_validation[0])


def compute_metrics(p):
    metric = load("squad_v2")
    predictions, labels = p
    start_preds = predictions[0].argmax(axis=1)
    end_preds = predictions[1].argmax(axis=1)

    formatted_predictions = [
        {
            "id": str(i),
            "prediction_text": tokenizer.decode(
                filtered_data_validation[i]["input_ids"][
                    start_preds[i] : end_preds[i] + 1
                ],  # Corrigido
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
                "text": [filtered_data_validation[i]["answer"]["text"]],
                "answer_start": [filtered_data_validation[i]["answer"]["offset"][0]],
            },
        }
        for i in range(len(filtered_data_validation))
    ]  # Corrigido

    return metric.compute(predictions=formatted_predictions, references=references)


data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="distilbert-qasports",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.01,
    push_to_hub=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=filtered_data_train,  # Processed training dataset
    eval_dataset=filtered_data_validation,  # Processed validation dataset
    data_collator=data_collator,  # Correct data collator
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

# Generate a simplified file name based on the selected DOC_READER and SPORT
save_name = f"{DOC_READER.name.lower()}-{SPORT.name.lower()}"

# Save tokenizer locally
# tokenizer.save_pretrained(save_name)

# Upload tokenizer to Hugging Face
# api = HfApi()
# api.upload_folder(
#     folder_path=save_name,
#     repo_id=f"laurafcamargos/{save_name}",
#     commit_message="Adding tokenizer",
# )

# Publish model to Hugging Face
# trainer.push_to_hub()

# Inference
question = "Who owns the building?"
context = "(See Memphis Summer Storm of 2003.) It was built at a cost of US$250 million and is owned by the City of Memphis, naming rights were purchased by one of Memphis' most well-known businesses, FedEx, for $92 million. FedExForum was financed using $250 million of public bonds, which were issued by the Memphis Public Building Authority (PBA)."

question_answerer = pipeline(
    task="question-answering", model=model, tokenizer=tokenizer
)
answer = question_answerer(question=question, context=context)

print(answer)
