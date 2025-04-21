from datasets import Dataset, load_dataset, load
import ast


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


# Certifique-se de definir seu token de acesso aqui ou como variável de ambiente
HUGGINGFACE_TOKEN = ""


def load_and_clean_data(data: Dataset) -> Dataset:
    """
    Converte 'answer' de string para dicionário e valida a estrutura.
    """
    cleaned_data = []
    for example in data:
        try:
            # Converter 'answer' de string para dicionário
            answer = (
                ast.literal_eval(example["answer"])
                if isinstance(example["answer"], str)
                else example["answer"]
            )
            context = example["context"]

            # Validação
            if (
                isinstance(answer, dict)
                and "text" in answer
                and "offset" in answer
                and isinstance(answer["offset"], list)
                and len(answer["offset"]) == 2
                and answer["offset"][1] > answer["offset"][0]
                and answer["offset"][1]
                <= len(context)  # Verifica se o offset está dentro do contexto
            ):
                cleaned_data.append(
                    {
                        "question": example["question"],
                        "context": context,
                        "answer": answer,  # Agora é um dicionário
                    }
                )
        except (SyntaxError, ValueError, KeyError):
            continue

    return Dataset.from_list(cleaned_data)


def preprocess_function(examples):
    questions = [
        q.strip() for q in examples["question"]
    ]  # Tokeniza as perguntas e os contextos
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",  # Trunca apenas o contexto
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answer"][i]

        # Fallback para respostas inválidas
        if not answer.get("text"):
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["offset"][0]
        end_char = answer["offset"][1]
        sequence_ids = inputs.sequence_ids(i)

        # Encontra o início e fim do contexto no texto tokenizado
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1

        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        # Se o contexto foi truncado, ajusta os limites
        tokenized_context_start = (
            offsets[context_start][0] if context_start < len(offsets) else 0
        )
        tokenized_context_end = offsets[context_end][1] if context_end >= 0 else 0

        # Verifica se a resposta está dentro do contexto tokenizado
        if start_char > tokenized_context_end or end_char < tokenized_context_start:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Ajusta os offsets para dentro do contexto truncado
            start_char = max(start_char, tokenized_context_start)
            end_char = min(end_char, tokenized_context_end)

            # Encontra os tokens
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
    return dataset.filter(
        lambda x: x["start_positions"]
        != 0  # remove todos os exemplos onde {'text': '', 'offset': [0, 0]}
        and x["end_positions"] != 0
        and x["start_positions"] <= x["end_positions"]  # garante que o início <= fim
    )


def dataset_load(split: str, sport: str):
    return load_dataset(
        "PedroCJardim/QASports", sport, split=split
    )  # Load and return dataset


def compute_metrics(p):
    metric = load("squad_v2")
    predictions, labels = p
    start_preds = predictions[0].argmax(axis=1)
    end_preds = predictions[1].argmax(axis=1)

    formatted_predictions = [
        {
            "id": str(i),
            "prediction_text": tokenizer.decode(
                filtered_data2[i]["input_ids"][
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
                "text": [filtered_data2[i]["answer"]["text"]],
                "answer_start": [filtered_data2[i]["answer"]["offset"][0]],
            },
        }
        for i in range(len(filtered_data2))
    ]  # Corrigido

    return metric.compute(predictions=formatted_predictions, references=references)
