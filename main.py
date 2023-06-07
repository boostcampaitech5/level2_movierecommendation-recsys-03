import hydra
from src.config import Config
from src.utils import get_timestamp


def main(config: Config = None) -> None:
    config.timestamp = get_timestamp()
    pass


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main_hydra(config: Config) -> None:
    main(config)


if __name__ == "__main__":
    main_hydra()
