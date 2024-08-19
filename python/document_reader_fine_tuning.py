# -*- coding: utf-8 -*-
"""Document_Reader_Fine_tuning.ipynb"""

import os
import enum
import ast

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


# Função para converter string para dicionário
def string_to_dict(string):
    try:
        return ast.literal_eval(string)
    except (SyntaxError, ValueError):
        print(f"Error evaluating string: {string}")
        return {}


# Função para limpar o dataset(antiga)
"""def clean_dataset(data: Dataset) -> Dataset:
    cleaned_data = []
    for tt in data:        
        answer = string_to_dict(tt["answer"])
        if answer["text"] and answer["offset"][1] != 0:
            cleaned_data.append(tt)
    return Dataset.from_list(cleaned_data)
"""


# Função para limpar o dataset(nova)
def clean_dataset(data: Dataset) -> Dataset:
    return data.filter(
        lambda example: string_to_dict(example["answer"])["text"]
        and string_to_dict(example["answer"])["offset"][1] != 0
    )


def load_and_clean_data(sport=SPORT):
    # Carregar dataset
    train = load_dataset("PedroCJardim/QASports", sport, split="train")
    test = load_dataset("PedroCJardim/QASports", sport, split="validation")
    
    # Limpar datasets
    cleaned_train = clean_dataset(train)
    cleaned_test = clean_dataset(test)
    
    # Criar DatasetDict
    dataset = DatasetDict({"train": cleaned_train, "test": cleaned_test})
    
    
    return dataset

# Tokenizador
tokenizer = AutoTokenizer.from_pretrained(DOC_READER)  

# carrega o dataset
dataset = load_and_clean_data() 

# Função de pré-processamento(antiga)
"""def preprocess_function(examples):
    examples["answer"] = [string_to_dict(t) for t in examples["answer"]]
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answer"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        if not answers[i]:
            start_positions.append(0)
            end_positions.append(0)
            continue
        answer = answers[i]
        start_char = answer["offset"][0]
        end_char = answer["offset"][1]
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
"""


# Função de pré-processamento(nova)
def preprocess_function(examples):
    examples["answer"] = [string_to_dict(t) for t in examples["answer"]]
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answer"]
    start_positions = []
    end_positions = []

    for i, (offset, answer) in enumerate(zip(offset_mapping, answers)):
        if not answer:
            # Sem resposta, definir posições como 0
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["offset"][0]
        end_char = answer["offset"][1]
        sequence_ids = inputs.sequence_ids(i)

        # Encontrar os limites do contexto
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1

        # Verificar se a resposta está completamente fora do intervalo
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Encontrar o índice do início da resposta
            start_idx = next(
                (idx for idx in range(context_start, context_end + 1)
                 if offset[idx][0] <= start_char <= offset[idx][1]),
                context_start  # Valor padrão caso não encontre
            )
            start_positions.append(start_idx)

            # Encontrar o índice do final da resposta
            end_idx = next(
                (idx for idx in range(context_end, context_start - 1, -1)
                 if offset[idx][0] <= end_char <= offset[idx][1]),
                context_end  # Valor padrão caso não encontre
            )
            end_positions.append(end_idx)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# Chama o pré-processamento apenas na coluna de test
tokenized_squad = dataset.map(
    preprocess_function, batched=True, remove_columns=dataset["train"].column_names
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
#trainer.push_to_hub()

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
