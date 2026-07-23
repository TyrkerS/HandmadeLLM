"""Config dataclasses + YAML loading. One config file per experiment, versioned in git."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4          # GQA: < n_heads shares KV heads across query groups
    vocab_size: int = 16384
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_weights: bool = True
    ffn_multiple_of: int = 64    # SwiGLU hidden dim rounded up to a multiple of this


@dataclass
class TrainConfig:
    # data
    train_bin: str = "data/tinystories/train.bin"
    val_bin: str = "data/tinystories/val.bin"
    tokenizer_path: str = "data/tinystories/tokenizer.json"
    # optimization
    batch_size: int = 32               # micro-batch (per forward pass)
    grad_accum_steps: int = 8          # effective batch = batch_size * grad_accum_steps
    max_steps: int = 20000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # efficiency
    dtype: str = "bfloat16"            # "bfloat16" | "float16" | "float32"
    compile: bool = False
    grad_checkpointing: bool = False
    # bookkeeping
    out_dir: str = "checkpoints/run"
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10
    seed: int = 1337
    wandb_project: str = "handmade-llm"
    run_name: str = "run"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            model=_build(ModelConfig, raw.get("model", {})),
            train=_build(TrainConfig, raw.get("train", {})),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, d: dict):
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**d)
