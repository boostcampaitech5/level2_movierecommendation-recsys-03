import os
import ray
import argparse
import numpy as np
from src.trainer import objective_function, get_scheduler, get_result
from src.utils import read_config, read_yaml
from src.predict import predict_to_submit
from src.preprocess import check_and_create
from functools import partial
from ray import tune
from ray.tune import CLIReporter


def main(args) -> None:
    np.float = np.float_

    config_file = os.path.join(args.base_dir, f"{args.config}.yaml")
    config = read_yaml(config_file)

    check_and_create(config)

    ray_config = os.path.join(args.base_dir, f"raytune.yaml")
    ray_config = read_config(ray_config)

    ray.init()
    reporter = CLIReporter(metric_columns=["recall"])
    scheduler = get_scheduler()

    tuner = tune.run(
        partial(objective_function, config_file_list=[config_file]),
        config=ray_config,
        num_samples=2,
        progress_reporter=reporter,
        scheduler=scheduler,
        resources_per_trial={"gpu": 1},
        keep_checkpoints_num=1,
        checkpoint_score_attr="recall",
    )

    best_trial = tuner.get_best_trial("recall", "max", "last")
    print(tuner.get_best_checkpoint(best_trial, "recall", "max"))
    best_config = best_trial.config

    model_name, best_valid_score, _, test_result, model_file = get_result([config_file], best_config)

    predict_to_submit(model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument
    arg("--config", type=str, default="default", help="config 파일의 버전을 설정할 수 있습니다.")
    arg("--base_dir", type=str, default="/opt/ml/input/level2_movierecommendation-recsys-03/recbole/configs")
    args = parser.parse_args()

    main(args)
