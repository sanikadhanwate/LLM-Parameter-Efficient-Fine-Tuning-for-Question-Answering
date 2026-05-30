import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from peft import PeftModel, PeftConfig
from src.config import cfg

def load_pipeline():
    config = PeftConfig.from_pretrained(cfg.output_dir)
    base = AutoModelForQuestionAnswering.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, cfg.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.output_dir)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

def answer_question(context: str, question: str):
    pipe = load_pipeline()
    result = pipe(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer:   {result['answer']}")
    print(f"Score:    {result['score']:.4f}")
    return result

if __name__ == "__main__":
    ctx = """
    Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA allow large language
    models to be fine-tuned at a fraction of the cost. LoRA injects trainable
    low-rank matrices into transformer layers while keeping the base model frozen.
    This reduces trainable parameters by over 99% compared to full fine-tuning.
    """
    answer_question(ctx, "What does LoRA inject into transformer layers?")