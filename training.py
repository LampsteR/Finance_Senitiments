from pathlib import Path
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from accelerate.state import AcceleratorState


class FinSentimentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_bundle(path: Path):
    return torch.load(path)


def build_model(num_labels: int):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"],
        task_type="SEQ_CLS",
    )

    return get_peft_model(model, lora_config)


def train(
    bundle_path: Path,
    output_dir: Path,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
):
    bundle = load_bundle(bundle_path)
    dataset = FinSentimentDataset(
        bundle["input_ids"], bundle["attention_mask"], bundle["labels"]
    )
    num_labels = bundle["labels"].unique().numel()

    model = build_model(num_labels)

    AcceleratorState._reset_state()
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_ratio=0.03,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    print(f"Training complete; checkpoints in {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    default_bundle = project_root / "artifacts" / "finbert_inputs.pt"
    default_output = project_root / "artifacts" / "finbert-qlora-checkpoints"
    default_output.mkdir(parents=True, exist_ok=True)
    train(default_bundle, default_output)

