import os
from utils import DocReader, Sports
from utils import (
    preprocess_function,
    load_and_clean_data,
    filter_dataset,
    compute_metrics,
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
from huggingface_hub import HfApi, login
from datasets import load_dataset
from dotenv import load_dotenv

# Read environment variables from .env file
load_dotenv()

# DOC_READER specifies the pretrained document reader model to be used
DOC_READER = DocReader.DistilBERT

# SPORT is used to choose a specific sport category from the QASports dataset
SPORT = Sports.BASKETBALL
print(SPORT.name, DOC_READER.name)

HUGGINGFACE_TOKEN = os.getenv["HUGGINGFACE_TOKEN"]
login(token=HUGGINGFACE_TOKEN)

# Load dataset
dataset_train = load_dataset("train[:100]", SPORT)
dataset_validation = load_dataset("validation[:10]", SPORT)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(DOC_READER)
model = AutoModelForQuestionAnswering.from_pretrained(DOC_READER)

# Complete preprocessing pipeline
cleaned_data_train = load_and_clean_data(dataset_train)
cleaned_data_validation = load_and_clean_data(dataset_validation)
processed_data_train = cleaned_data.map(preprocess_function, batched=True)
processed_data_validation = cleaned_data_validation.map(
    preprocess_function, batched=True
)
filtered_data_train = filter_dataset(processed_data)
filtered_data_validation = filter_dataset(processed_data2)

# Checking the size of preprocessed vectors
print(
    f"Original: {len(dataset_train)}, Cleaned: {len(cleaned_data_train)}, Filtered: {len(filtered_data_train)}"
)
print(
    f"Original: {len(dataset_validation)}, Cleaned: {len(cleaned_data_validation)}, Filtered: {len(filtered_data_validation)}"
)

# Checking a sample vector
print(filtered_data_validation[0])

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
api = HfApi()
api.upload_folder(
    folder_path=save_name,
    repo_id=f"laurafcamargos/{save_name}",
    commit_message="Adding tokenizer",
)

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
