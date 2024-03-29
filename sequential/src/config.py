from dataclasses import dataclass
from omegaconf import ListConfig


@dataclass
class Data:
    data_version: str
    batch_size: int
    pre_batch_size: int
    max_seq_length: int
    pre_data_augmentation: bool
    prepare_data: bool
    mask_p: float
    item_size: int
    mask_id: int


@dataclass
class Path:
    train_dir: str
    eval_dir: str
    output_dir: str
    train_file: str
    eval_file: str
    attr_files: ListConfig
    pretrain_file: str
    idx2item_file: str


@dataclass
class Trainer:
    is_pretrain: bool
    use_pretrain: bool
    pretrain_version: str
    cv: bool
    k: int
    epoch: int
    lr: float
    weight_decay: float
    aap_weight: float
    mip_weight: float
    map_weight: float
    sp_weight: float
    adam_beta1: float
    adam_beta2: float


@dataclass
class Model:
    model_name: str
    hidden_size: int
    num_hidden_layers: int
    initializer_range: float
    num_attention_heads: int
    hidden_act: str
    attention_probs_dropout_prob: float
    hidden_dropout_prob: float
    item_size: int


@dataclass
class Wandb:
    project: str
    entity: str
    name: str


@dataclass
class Config:
    timestamp: str
    seed: int
    gpu_id: int
    no_cuda: bool
    cuda_condition: bool
    data: Data
    path: Path
    trainer: Trainer
    model: Model
    wandb: Wandb
