import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, RandomSampler
from src.config import Config
from src.utils import get_user_seqs_long, get_item2attr_json
from src.datasets import PretrainDataset


class PretrainDataModule(L.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

        # data_file = config.path.data_dir + config.path.data_name + '.txt'
        self.data_file = config.path.data_dir + "train_ratings.csv"
        self.item2attr_file = config.path.data_dir + config.data.data_name + "_item2attributes.json"

    def prepare_data(self) -> None:
        # concat all user_seq get a long sequence, from which sample neg segment for SP
        self.user_seq, self.max_item, self.long_seq = get_user_seqs_long(self.config.path.data_dir)
        self.item2attr, self.attr_size = get_item2attr_json(self.item2attr_file)

    def train_dataloader(self) -> DataLoader:
        trainset = PretrainDataset(self.config, self.user_seq, self.long_seq)
        sampler = RandomSampler(trainset)
        return DataLoader(trainset, sampler=sampler, batch_size=self.config.data.pre_batch_size)
