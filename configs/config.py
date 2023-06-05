from dataclasses import dataclass


@dataclass
class Data:
    data_name: str
    batch_size: int
    pre_batch_size: int
    max_seq_length: int
    mask_p: float


@dataclass
class Path:
    data_dir: str
    output_dir: str


@dataclass
class Trainer:
    epoch: int
    pre_epochs: int
    lr: float
    weight_decay: float
    log_freq: int
    aap_weight: float
    mip_weight: float
    map_weight: float
    sp_weight: float
    adam_beta1: float
    adam_beta2: float


@dataclass
class Model:
    model_name: str


@dataclass
class Config:
    seed: int
    gpu_id: int
    no_cuda: bool
    data: Data
    path: Path
    trainer: Trainer
    model: Model


@dataclass
class S3Rec(Model):
    hidden_size: int
    num_hidden_layers: int
    initializer_range: float
    num_attention_heads: int
    hidden_act: str
    attention_probs_dropout_prob: float
    hidden_dropout_prob: float
    item_size: int
