import os
import ray
import wandb
import argparse
import numpy as np
from ray import tune
from functools import partial
from ray.tune import CLIReporter
from src.predict import predict_to_submit
from src.trainer import objective_function, get_scheduler
from src.preprocess import check_and_create_inter, check_and_create_item
from src.utils import read_config, read_yaml, init_wandb, get_result, set_path, get_timestamp


def main(args) -> None:
    # Basic Settings
    np.float = np.float_
    base_path = os.getcwd()
    timestamp = get_timestamp()

    # Get Config Files
    config_file = os.path.join(base_path, f"configs/{args.config}.yaml")
    ray_config = os.path.join(base_path, f"configs/raytune.yaml")

    # Read Configs
    config = read_yaml(config_file)
    ray_config = read_config(ray_config)

    # Create Data for RecBole
    set_path(config, base_path)
    check_and_create_inter(config)
    check_and_create_item(config)

    # Settings for Ray Tune
    ray.init()
    scheduler = get_scheduler()
    reporter = CLIReporter(metric_columns=["recall"], print_intermediate_tables=True, max_report_frequency=600)

    # Running ray tune
    tuner = tune.run(
        partial(objective_function, config_dict=config),
        config=ray_config,
        num_samples=config["count"],
        progress_reporter=reporter,
        scheduler=scheduler,
        resources_per_trial={"gpu": 1},
    )

    # WandB Logging
    init_wandb(config, timestamp)

    best_config, best_model = get_result(tuner)
    predict_to_submit(best_model, timestamp)
    print(tuner.results_df)

    wandb.save(best_model)
    wandb.log(best_config)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument
    arg("--config", type=str, default="default", help="config 파일의 버전을 설정할 수 있습니다.")
    args = parser.parse_args()

    main(args)
