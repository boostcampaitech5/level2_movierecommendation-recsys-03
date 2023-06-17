import json
import math
import os
import random
import wandb
import dotenv
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.sparse import csr_matrix
from src.config import Config


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def get_timestamp(date_format: str = "%d_%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)


def init_wandb(is_pretrain: bool, config: Config) -> None:
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)

    if is_pretrain:
        run = wandb.init(
            project=config.wandb.project + "Pretrain",
            entity=config.wandb.entity,
            name=config.wandb.name,
        )
    else:
        run = wandb.init(
            project=config.wandb.project + "Sequential",
            entity=config.wandb.entity,
            name=config.wandb.name,
        )
        run.tags = [config.model.model_name]


def log_parameters(is_pretrain: bool, config: Config) -> None:
    # model config
    wandb.log(
        {
            "hidden_size": config.model.hidden_size,
            "num_hidden_layers": config.model.num_hidden_layers,
            "initializer_range": config.model.initializer_range,
            "num_attention_heads": config.model.num_attention_heads,
            "hidden_act": config.model.hidden_act,
            "attention_probs_dropout_prob": config.model.attention_probs_dropout_prob,
            "hidden_dropout_prob": config.model.hidden_dropout_prob,
        }
    )
    # common trainer config
    wandb.log(
        {
            "lr": config.trainer.lr,
            "weight_decay": config.trainer.weight_decay,
            "adam_beta1": config.trainer.adam_beta1,
            "adam_beta2": config.trainer.adam_beta2,
            "max_seq_length": config.data.max_seq_length,
            "mask_p": config.data.mask_p,
        }
    )
    if is_pretrain:
        # pretrain config
        wandb.log(
            {
                "aap_weight": config.trainer.aap_weight,
                "mip_weight": config.trainer.mip_weight,
                "map_weight": config.trainer.map_weight,
                "sp_weight": config.trainer.sp_weight,
                "batch_size": config.data.pre_batch_size,
            }
        )
    else:
        # finetune config
        wandb.log(
            {
                "use_pretrain": config.trainer.use_pretrain,
                "pretrain_version": config.trainer.pretrain_version,
                "cv": config.trainer.cv,
                "k": config.trainer.k,
                "batch_size": config.data.batch_size,
            }
        )


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"{dir} created")


def neg_sample(item_set: set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_submission_file(config: Config, preds):
    sample_sub_path = os.path.join(config.path.eval_dir, config.path.eval_file)

    check_dir(config.path.output_dir)

    sub_path = os.path.join(config.path.output_dir, f"{config.timestamp}_{config.model.model_name}_submit.csv")

    sub_df = pd.read_csv(sample_sub_path)

    items = np.vstack(preds)
    items = items.reshape(-1)

    sub_df.loc[:, "item"] = items
    sub_df.to_csv(sub_path, index=False)
    wandb.save(sub_path)


def get_user_seqs(train_dir, train_file):
    data_path = os.path.join(train_dir, train_file)

    rating_df = pd.read_csv(data_path)
    lines = rating_df.groupby("user")["item"].apply(list)
    user_seq = []
    item_set = set()
    for line in lines:
        items = line
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    submission_rating_matrix = generate_rating_matrix_submission(user_seq, num_users, num_items)
    return (
        user_seq,
        max_item,
        valid_rating_matrix,
        test_rating_matrix,
        submission_rating_matrix,
    )


def get_user_seqs_long(train_dir, train_file):
    data_path = os.path.join(train_dir, train_file)

    rating_df = pd.read_csv(data_path)
    lines = rating_df.groupby("user")["item"].apply(list)
    user_seq = []
    long_seq = []
    item_set = set()
    for line in lines:
        items = line
        long_seq.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_seq


def get_item2attr_json(train_dir, attr_file):
    data_path = os.path.join(train_dir, attr_file)

    with open(data_path) as f:
        item2attr = json.loads(f.readline())

    item2attr = {int(key): value for key, value in item2attr.items()}

    attr_set = set()
    for item_id, attrs in item2attr.items():
        attr_set = attr_set | set(attrs)
    attr_size = max(attr_set)
    return item2attr, attr_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
