import json
from transformers import AutoTokenizer

TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128
TRAIN_PATH = "raw_data/train-v2.0.json"
DEV_PATH   = "raw_data/dev-v2.0.json"

def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def load_squad_json(path):
    with open(path) as f:
        raw = json.load(f)
    examples = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({
                    "id":       qa["id"],
                    "question": qa["question"],
                    "context":  context,
                    "answers":  {
                        "text":         [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]]
                    }
                })
    return examples

def load_squad():
    print("Loading SQuAD v2 from local JSON...")
    train = load_squad_json(TRAIN_PATH)
    val   = load_squad_json(DEV_PATH)
    print(f"Train: {len(train)} | Val: {len(val)}")
    return {"train": train, "validation": val}

def tokenize_examples(examples, tokenizer):
    tokenized = {"input_ids": [], "attention_mask": [],
                 "token_type_ids": [], "start_positions": [], "end_positions": []}

    for ex in examples:
        inputs = tokenizer(
            ex["question"],
            ex["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        inputs.pop("overflow_to_sample_mapping")

        for i, offset in enumerate(offset_mapping):
            sequence_ids = inputs.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            answers = ex["answers"]
            if len(answers["answer_start"]) == 0:
                start_pos, end_pos = 0, 0
            else:
                start_char = answers["answer_start"][0]
                end_char   = start_char + len(answers["text"][0])
                if (offset[context_start][0] > start_char or
                        offset[context_end][1] < end_char):
                    start_pos, end_pos = 0, 0
                else:
                    j = context_start
                    while j <= context_end and offset[j][0] <= start_char:
                        j += 1
                    start_pos = j - 1
                    j = context_end
                    while j >= context_start and offset[j][1] >= end_char:
                        j -= 1
                    end_pos = j + 1

            tokenized["input_ids"].append(inputs["input_ids"][i])
            tokenized["attention_mask"].append(inputs["attention_mask"][i])
            ttype = inputs.get("token_type_ids")
            tokenized["token_type_ids"].append(
            ttype[i] if ttype else [0] * MAX_LENGTH)
            tokenized["start_positions"].append(start_pos)
            tokenized["end_positions"].append(end_pos)

    return tokenized

import torch
from torch.utils.data import Dataset

class SQuADDataset(Dataset):
    def __init__(self, tokenized):
        self.input_ids      = torch.tensor(tokenized["input_ids"])
        self.attention_mask = torch.tensor(tokenized["attention_mask"])
        self.token_type_ids = torch.tensor(tokenized["token_type_ids"])
        self.start_positions= torch.tensor(tokenized["start_positions"])
        self.end_positions  = torch.tensor(tokenized["end_positions"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids":       self.input_ids[idx],
            "attention_mask":  self.attention_mask[idx],
            "token_type_ids":  self.token_type_ids[idx],
            "start_positions": self.start_positions[idx],
            "end_positions":   self.end_positions[idx],
        }

def get_tokenized_dataset(tokenizer, small=False):
    data = load_squad()
    train_ex = data["train"][:2000] if small else data["train"]
    val_ex   = data["validation"][:500] if small else data["validation"]

    print("Tokenizing train...")
    train_tok = tokenize_examples(train_ex, tokenizer)
    print("Tokenizing validation...")
    val_tok   = tokenize_examples(val_ex, tokenizer)

    return {
        "train":      SQuADDataset(train_tok),
        "validation": SQuADDataset(val_tok)
    }

if __name__ == "__main__":
    tok = get_tokenizer()
    ds  = get_tokenized_dataset(tok, small=True)
    print(f"Train samples: {len(ds['train'])}")
    print(f"Val samples:   {len(ds['validation'])}")
    print("Sample keys:", list(ds["train"][0].keys()))
    print("Done.")