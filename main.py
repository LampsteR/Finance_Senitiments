import argparse
from pathlib import Path

from tokenising import tokenize_dataset
from training import train
from run import load_model, classify_sentiment


def main():
    parser = argparse.ArgumentParser(description="Financial sentiment workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize raw JSONL data")
    tokenize_parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent
        / "datasets"
        / "merged_financial_sentiment.json",
    )
    tokenize_parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "finbert_inputs.pt",
    )
    tokenize_parser.add_argument("--model-name", default="ProsusAI/finbert")
    tokenize_parser.add_argument("--max-length", type=int, default=256)

    train_parser = subparsers.add_parser("train", help="Train QLoRA adapter")
    train_parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "finbert_inputs.pt",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "finbert-qlora-checkpoints",
    )
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--grad-accum", type=int, default=2)

    run_parser = subparsers.add_parser("run", help="Run inference with trained adapter")
    run_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "finbert-qlora-checkpoints"
        / "checkpoint-latest",
    )
    run_parser.add_argument("--text", action="append", help="Text to score (can repeat)")

    args = parser.parse_args()

    if args.command == "tokenize":
        tokenize_dataset(args.input_jsonl, args.output, args.model_name, args.max_length)

    elif args.command == "train":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        train(
            bundle_path=args.bundle,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
        )

    elif args.command == "run":
        tokenizer, model = load_model(args.checkpoint)
        texts = args.text or ["The company posted record profits and raised guidance."]
        for text, label, confidence in classify_sentiment(texts, tokenizer, model):
            print(f"Text: {text}\nPrediction: {label} (confidence={confidence:.3f})\n")


if __name__ == "__main__":
    main()
