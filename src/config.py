from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "squad_v2"
    output_dir: str = "results/peft-qa-model"

    # LoRA hyperparameters
    lora_r: int = 8           # rank — higher = more params, more expressive
    lora_alpha: int = 16      # scaling factor (alpha / r = scaling)
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("q_lin", "k_lin", "v_lin", "out_lin")

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Data
    max_length: int = 384
    doc_stride: int = 128
    use_small_dataset: bool = True  # set False for full run

    # Misc
    seed: int = 42
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500

cfg = TrainingConfig()