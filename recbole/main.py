import os
import ray
import wandb
import argparse
import numpy as np
from src.trainer import objective_function, get_scheduler
from src.utils import read_config, read_yaml, init_wandb, get_result
from src.predict import predict_to_submit
from src.preprocess import check_and_create_data
from functools import partial
from ray import tune
from ray.tune import CLIReporter


def main(args) -> None:
    np.float = np.float_

    config_file = os.path.join(args.base_dir, f"{args.config}.yaml")
    config = read_yaml(config_file)

    check_and_create_data(config)
    ray_config = os.path.join(args.base_dir, f"raytune.yaml")
    ray_config = read_config(ray_config)

    ray.init()
    reporter = CLIReporter(metric_columns=["recall"])
    scheduler = get_scheduler()

    tuner = tune.run(
        partial(objective_function, config_file_list=[config_file]),
        config=ray_config,
        num_samples=config["count"],
        progress_reporter=reporter,
        scheduler=scheduler,
        resources_per_trial={"gpu": 1},
        keep_checkpoints_num=1,
        checkpoint_score_attr="recall",
    )

    init_wandb(config)

    best_config, best_model = get_result(tuner)
    predict_to_submit(best_model)
    print(tuner.results_df)

    wandb.save(best_model)
    wandb.log(best_config)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument
    arg("--config", type=str, default="default", help="config 파일의 버전을 설정할 수 있습니다.")
    arg("--base_dir", type=str, default="/opt/ml/input/level2_movierecommendation-recsys-03/recbole/configs")
    args = parser.parse_args()

    main(args)
