import lightning as L
from torch.utils.data import DataLoader, RandomSampler
from src.config import Config
from src.utils import get_user_seqs_long, get_attrs_from_json
from src.datasets import S3RecDataset


class S3RecDataModule(L.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

        self.train_dir = config.path.train_dir
        self.train_file = config.path.train_file

        self.attr_names = list(config.path.attr_files.keys())
        self.attr_files = [config.data.data_version + "_" + config.path.attr_files[attr_name] for attr_name in self.attr_names]

    def prepare_data(self) -> None:
        # concat all user_seq get a long sequence, from which sample neg segment for SP
        self.user_seq, max_item, self.long_seq = get_user_seqs_long(self.train_dir, self.train_file)

        self.name2item2attr, self.name2attr_size = get_attrs_from_json(self.train_dir, self.attr_files, self.attr_names)

        self.config.data.item_size = max_item + 2
        self.config.data.mask_id = max_item + 1

    def train_dataloader(self) -> DataLoader:
        trainset = S3RecDataset(self.config, self.user_seq, self.long_seq, self.name2item2attr, self.name2attr_size)
        sampler = RandomSampler(trainset)
        return DataLoader(trainset, sampler=sampler, batch_size=self.config.data.pre_batch_size, num_workers=0)
