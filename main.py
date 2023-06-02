import hydra
from configs.config import Config


def main(config: Config = None) -> None:
    pass


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main_hydra(config: Config) -> None:
    main(config)


if __name__ == "__main__":
    main_hydra()
