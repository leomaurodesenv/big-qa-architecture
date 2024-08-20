# -*- coding: utf-8 -*-
"""Document_Reader_Fine_tuning.ipynb"""

import os
import enum
import ast
from utils import (
    load_and_clean_data,
    preprocess_function,
    string_to_dict,
    filter_dataset,
)
from datasets import load_dataset, Dataset, DatasetDict
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


class DocReader:
    """Document Reader options"""

    BERT = "deepset/bert-base-uncased-squad2"
    RoBERTa = "deepset/roberta-base-squad2"
    MiniLM = "deepset/minilm-uncased-squad2"
    DistilBERT = "distilbert-base-uncased-distilled-squad"
    ELECTRA = "deepset/electra-base-squad2"
    SmallDistilBERT = "laurafcamargos/distilbert-qasports-basket-small"


class Sports:
    BASKETBALL = "basketball"
    FOOTBALL = "football"
    SOCCER = "soccer"
    ALL = ""


DOC_READER = DocReader.DistilBERT
SPORT = Sports.BASKETBALL
HUGGINGFACE_TOKEN = "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"


os.environ["HUGGINGFACE_TOKEN"] = HUGGINGFACE_TOKEN
login(token=HUGGINGFACE_TOKEN)


# Tokenizador
tokenizer = AutoTokenizer.from_pretrained(DOC_READER)

# carrega o dataset
dataset = load_and_clean_data(SPORT)


# Chama o pré-processamento apenas na coluna de test
tokenized_squad = dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Data collator
data_collator = DefaultDataCollator()

# Carregar modelo
model = AutoModelForQuestionAnswering.from_pretrained(DOC_READER)

# Argumentos de treinamento
training_args = TrainingArguments(
    output_dir="distilbert-qasports-basketball",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1000,
    weight_decay=0.01,
    push_to_hub=True,
    load_best_model_at_end=True,
)

# Configurar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Treina o modelo
trainer.train()

# Publicar modelo no HuggingFace (descomente para usar)
# trainer.push_to_hub()

# Inference
question = "What team waived Jones on October 25?"
context = (
    "Sacramento Kings (2007-2008) On September 27, 2007, Jones signed with the Boston Celtics. "
    "However, he was later waived by the Celtics on October 25. On December 10, he signed with the "
    "Sacramento Kings. Four days later, he made his debut with the Kings in a 109-99 win over the "
    "Philadelphia 76ers, recording one assist and two steals in seven minutes off the bench."
)

question_answerer = pipeline(
    task="question-answering", model=model, tokenizer=tokenizer
)
answer = question_answerer(question=question, context=context)

print(answer)
