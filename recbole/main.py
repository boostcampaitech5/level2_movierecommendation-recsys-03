import os
import argparse
import numpy as np
import pandas as pd
from logging import getLogger
from datetime import datetime
from src.models.srgnn import SRGNN
from src.preprocess import check_existence, create_inter
from recbole.config import Config
from recbole.trainer.trainer import Trainer
from recbole.utils import init_logger, init_seed
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
from recbole.data import create_dataset, data_preparation


def main(args) -> None:
    np.float = np.float_
    logger = getLogger()

    data_file, existence = check_existence()
    if existence == False:
        train = pd.read_csv("../data/train/train_ratings.csv")
        create_inter(data_file, train)

    config = Config(model="SRGNN", dataset=args.mode, config_file_list=["./configs/default.yaml"])
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)

    logger.info(config)

    dataset = create_dataset(config)

    logger.info(dataset)

    print("-------------------- Preparing Dataloaders --------------------")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = SRGNN(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    print("-------------------- Start training --------------------")
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])

    print(f"best_valid_score :{best_valid_score}\nbest_valid_result :{best_valid_result}")
    print("-------------------- Load models for prediction --------------------")

    config, model, dataset, _, _, test_data = load_data_and_model(
        model_file=trainer.saved_model_file,
    )

    external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]
    uid_series = dataset.token2id(dataset.uid_field, external_user_ids)

    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config["device"])
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()).flatten()

    if args.mode == "train":
        sub_file = "sample_submission.csv"
    else:
        sub_file = "dummy.csv"

    sub_path = os.path.join(config["submission_path"], sub_file)
    submission = pd.read_csv(sub_path)
    submission["item"] = external_item_list

    timestamp = datetime.now()
    timestamp = timestamp.strftime("%d_%H%M%S")
    print("-------------------- Saving results --------------------")

    os.makedirs(config["output_path"], exist_ok=True)
    out_path = os.path.join(config["output_path"], f"{args.mode}:{timestamp}_{config['model']}_submit.csv")
    submission.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument
    arg("--mode", type=str, default="train", choices=["dummy", "train"], help="running mode를 선택할 수 있습니다.")
    args = parser.parse_args()

    main(args)
