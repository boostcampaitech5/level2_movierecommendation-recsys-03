import os
import wandb
import pandas as pd
from .utils import get_timestamp
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model


def predict_to_submit(model_file: str) -> None:
    print("-------------------- Load models for prediction --------------------")

    config, model, dataset, _, _, test_data = load_data_and_model(
        model_file=model_file,
    )

    external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]
    uid_series = dataset.token2id(dataset.uid_field, external_user_ids)

    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config["device"])
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()).flatten()

    sub_file = "sample_submission.csv"
    sub_path = os.path.join(config["submission_path"], sub_file)
    submission = pd.read_csv(sub_path)
    submission["item"] = external_item_list

    timestamp = get_timestamp()
    print("-------------------- Saving results --------------------")

    os.makedirs(config["output_path"], exist_ok=True)
    out_path = os.path.join(config["output_path"], f"{timestamp}_{config['model']}_submit.csv")
    submission.to_csv(out_path, index=False)
    wandb.save(out_path)
    print("-------------------- Submission saved --------------------")
