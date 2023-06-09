import os
import hydra
from omegaconf import DictConfig


def __main(config: DictConfig = None) -> None:
    pass


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: DictConfig = None) -> None:
    __main(config)


if __name__ == "__main__":
    main()
