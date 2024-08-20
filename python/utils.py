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


# Função para converter string para dicionário
def string_to_dict(string):
    try:
        return ast.literal_eval(string)
    except (SyntaxError, ValueError):
        print(f"Error evaluating string: {string}")
        return {}

# Função para limpar o dataset(nova)
def filter_dataset(data: Dataset) -> Dataset:
    return data.filter(
        lambda example: string_to_dict(example["answer"])["text"]
        and string_to_dict(example["answer"])["offset"][1] != 0
    )

def load_and_clean_data(sport):
    # Carregar dataset
    train = load_dataset("PedroCJardim/QASports", sport, split="train")
    test = load_dataset("PedroCJardim/QASports", sport, split="validation")
    
    # Limpar datasets
    cleaned_train = filter_dataset(train)
    cleaned_test = filter_dataset(test)
    
    # Criar DatasetDict
    dataset = DatasetDict({"train": cleaned_train, "test": cleaned_test})
    
    
    return dataset

# Função de pré-processamento(nova)
def preprocess_function(examples,tokenizer):
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

