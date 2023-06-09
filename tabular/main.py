import os
import wandb
import hydra
import dotenv
from omegaconf import DictConfig

from tabular.utils import set_seeds, get_timestamp


def __main(config: DictConfig = None) -> None:
    # setting
    print(f"----------------- Setting -----------------")
    config.timestamp = get_timestamp()
    config.wandb.name = f"work-{get_timestamp()}"
    set_seeds(config.seed)

    # wandb init
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
    )

    wandb.finish()


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None) -> None:
    __main(config)


if __name__ == "__main__":
    main()
