import torch
import json
import os
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from src.config import cfg
from src.model import get_model_and_tokenizer
from data.prepare_data import get_tokenized_dataset

def train():
    print("=== PEFT LoRA Fine-tuning for QA ===")
    torch.manual_seed(cfg.seed)

    model, tokenizer = get_model_and_tokenizer()
    dataset = get_tokenized_dataset(tokenizer, small=cfg.use_small_dataset)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=cfg.per_device_eval_batch_size
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    total_steps = len(train_loader) * cfg.num_train_epochs
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on: {device}")

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.num_train_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % cfg.logging_steps == 0 and step > 0:
                avg = total_loss / (step + 1)
                print(f"  Step {step} | Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch+1} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(cfg.output_dir, exist_ok=True)
            model.save_pretrained(cfg.output_dir)
            tokenizer.save_pretrained(cfg.output_dir)
            print(f"  Model saved to {cfg.output_dir}")

    os.makedirs("results", exist_ok=True)
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete. History saved to results/training_history.json")
    return history

if __name__ == "__main__":
    train()