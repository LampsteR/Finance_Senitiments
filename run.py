from pathlib import Path
from typing import Iterable, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

LABEL_NAMES = ["negative", "neutral", "positive"]


def load_model(checkpoint_dir: Path):
    base_model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Try to use bitsandbytes if available, otherwise use regular loading
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(LABEL_NAMES),
            quantization_config=bnb_config,
            device_map="auto",
        )
    except (ImportError, Exception):
        # Fallback to regular loading without quantization
        print("Warning: bitsandbytes not available, loading model without quantization")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(LABEL_NAMES),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    return tokenizer, model


def classify_sentiment(
    texts: Iterable[str],
    tokenizer,
    model,
    max_length: int = 256,
) -> List[Tuple[str, str, float]]:
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1).cpu().tolist()
        confidences = probs.max(dim=-1).values.cpu().tolist()

    results = []
    for text, idx, confidence in zip(texts, pred_ids, confidences):
        label = LABEL_NAMES[idx]
        results.append((text, label, confidence))
    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    default_checkpoint = (
        project_root / "artifacts" / "finbert-qlora-checkpoints" / "checkpoint-latest"
    )

    if not default_checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {default_checkpoint}\n"
            "Provide an explicit path or train a model first."
        )

    tokenizer, model = load_model(default_checkpoint)
    samples = [
        "The company posted record profits and raised guidance.",
        "Regulators opened an investigation into the firm's accounting practices.",
        "Management expects performance to remain stable next quarter.",
    ]

    for text, label, confidence in classify_sentiment(samples, tokenizer, model):
        print(f"Text: {text}\nPrediction: {label} (confidence={confidence:.3f})\n")

