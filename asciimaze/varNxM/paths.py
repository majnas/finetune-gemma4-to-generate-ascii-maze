from pathlib import Path

BASE_DIR = Path(__file__).parent


def experiment_name(base_name: str, cfg: dict) -> str:
    """Encode the training hyperparameters that distinguish one run from
    another for the same base model, so multiple experiments don't
    collide."""
    max_seq_length = cfg.get("max_seq_length", 1024)
    num_train_epochs = cfg.get("num_train_epochs", 3)
    per_device_train_batch_size = cfg.get("per_device_train_batch_size", 8)
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 2)

    return (
        f"{base_name}"
        f"_mslen_{max_seq_length}"
        f"_epochs_{num_train_epochs}"
        f"_bs_{per_device_train_batch_size}x{gradient_accumulation_steps}"
    )


def experiment_dir(base_name: str, cfg: dict) -> Path:
    return BASE_DIR / "outputs" / base_name / experiment_name(base_name, cfg)
