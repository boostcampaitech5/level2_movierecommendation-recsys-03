from dataclasses import dataclass


@dataclass
class Data:
    data_version: str
    batch_size: int
    pre_batch_size: int
    max_seq_length: int
    mask_p: float
    item_size: int
    mask_id: int
    attr_size: int
    item2attr: dict


@dataclass
class Path:
    train_dir: str
    eval_dir: str
    output_dir: str
    train_file: str
    eval_file: str
    attr_file: str
    pretrain_file: str


@dataclass
class Trainer:
    is_pretrain: bool
    use_pretrain: bool
    pretrain_version: str
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
