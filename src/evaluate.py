import json
import torch
import collections
import evaluate as hf_evaluate
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from src.config import cfg

def load_trained_model():
    config = PeftConfig.from_pretrained(cfg.output_dir)
    from transformers import AutoModelForQuestionAnswering
    base = AutoModelForQuestionAnswering.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, cfg.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.output_dir)
    model.eval()
    return model, tokenizer

def get_predictions(model, tokenizer, examples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions = {}

    for example in tqdm(examples, desc="Predicting"):
        inputs = tokenizer(
            example["question"],
            example["context"],
            max_length=cfg.max_length,
            truncation="only_second",
            stride=cfg.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        offsets = inputs.pop("offset_mapping")
        inputs.pop("overflow_to_sample_mapping")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs.start_logits.cpu()
        end_logits = outputs.end_logits.cpu()

        # Best span
        best_score = float("-inf")
        best_answer = ""
        for i in range(len(start_logits)):
            offset = offsets[i]
            for s in torch.argsort(start_logits[i], descending=True)[:5]:
                for e in torch.argsort(end_logits[i], descending=True)[:5]:
                    if e < s or e - s > 30:
                        continue
                    score = start_logits[i][s] + end_logits[i][e]
                    if score > best_score:
                        best_score = score
                        start_char = offset[s][0].item()
                        end_char = offset[e][1].item()
                        best_answer = example["context"][start_char:end_char]

        predictions[example["id"]] = best_answer

    return predictions

def evaluate_model():
    model, tokenizer = load_trained_model()
    dataset = load_dataset("squad_v2", split="validation").select(range(500))
    predictions = get_predictions(model, tokenizer, dataset)

    references = [
        {"id": ex["id"], "answers": ex["answers"]}
        for ex in dataset
    ]

    metric = hf_evaluate.load("squad_v2")
    results = metric.compute(
        predictions=[{"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                     for k, v in predictions.items()],
        references=references
    )

    print("\n=== Evaluation Results ===")
    print(f"Exact Match: {results['exact']:.2f}%")
    print(f"F1 Score:    {results['f1']:.2f}%")
    print(f"HasAns EM:   {results['HasAns_exact']:.2f}%")
    print(f"NoAns EM:    {results['NoAns_exact']:.2f}%")

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to results/metrics.json")
    return results

if __name__ == "__main__":
    evaluate_model()