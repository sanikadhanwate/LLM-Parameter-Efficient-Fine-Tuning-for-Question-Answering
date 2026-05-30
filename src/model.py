from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from src.config import cfg

def load_base_model():
    print(f"Loading base model: {cfg.model_name}")
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model parameters: {total_params:,}")
    return model, tokenizer

def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    pct = 100 * trainable / total
    print(f"Trainable parameters: {trainable:,} / {total:,}  ({pct:.2f}%)")
    return peft_model

def get_model_and_tokenizer():
    model, tokenizer = load_base_model()
    peft_model = apply_lora(model)
    return peft_model, tokenizer

if __name__ == "__main__":
    m, t = get_model_and_tokenizer()
    m.print_trainable_parameters()