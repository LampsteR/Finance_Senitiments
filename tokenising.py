from pathlib import Path
import json
import torch
from transformers import AutoTokenizer

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_label(raw_label):
    if isinstance(raw_label, str):
        return LABEL_TO_ID[raw_label.lower()]
    return int(raw_label)


def tokenize_dataset(
    input_path: Path,
    output_path: Path,
    model_name: str = "ProsusAI/finbert",
    max_length: int = 256,
):
    records = load_jsonl(input_path)
    texts = [record["text"] for record in records]
    labels = [normalize_label(record["label"]) for record in records]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    print(f"Saved tensors to {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    default_input = project_root / "datasets" / "merged_financial_sentiment.json"
    default_output = project_root / "artifacts" / "finbert_inputs.pt"
    tokenize_dataset(default_input, default_output)

