from ray import tune
from recbole.config import Config
from typing import Dict, Any, Optional, Tuple
from recbole.utils import init_logger, init_seed
from recbole.utils.utils import get_model, get_trainer
from recbole.data import create_dataset, data_preparation
from ray.tune.schedulers import ASHAScheduler


def get_scheduler() -> ASHAScheduler:
    scheduler = ASHAScheduler(metric="recall", mode="max", grace_period=1, reduction_factor=2, brackets=1, max_t=100)
    return scheduler


def train_recbole(config: dict = None) -> Tuple[str, float, dict[str, float], float, str]:
    config = Config(config_dict=config)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_name = config["model"]
    model = get_model(model_name)(config, train_data.dataset).to(config["device"])

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=config["show_progress"], verbose=False)
    test_result = trainer.evaluate(test_data)
    model_file = trainer.saved_model_file

    return model_name, best_valid_score, best_valid_result, test_result, model_file


def objective_function(config: None, config_dict: dict) -> Dict[str, Any]:
    model_name, best_valid_score, best_valid_result, test_result, _ = train_recbole(config=config_dict)
    tune.report(recall=best_valid_score)

    return {"model": model_name, "recall": best_valid_score, "best_valid_result/recall@10": best_valid_result, "test_result": test_result}
