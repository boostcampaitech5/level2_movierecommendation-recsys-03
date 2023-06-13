import hydra
import os
import omegaconf
from src.config import Config
from main import main
from unittest.mock import patch


@patch("wandb.init")
@patch("wandb.login")
@patch("wandb.log")
@patch("wandb.finish")
@patch("wandb.save")
def test_main(*args, **kwargs):
    @hydra.main(version_base="1.2", config_path="configs", config_name="test.yaml")
    def inner_main(config: Config) -> None:
        for model_path in os.listdir("configs/model.test"):
            model_path = os.path.join("configs/model.test", model_path)

            model_config = omegaconf.OmegaConf.load(model_path)
            config.model = model_config

            for trainer_path in sorted(os.listdir("configs/trainer.test")):
                trainer_path = os.path.join("configs/trainer.test", trainer_path)

                trainer_config = omegaconf.OmegaConf.load(trainer_path)
                config.trainer = trainer_config

                main(config)

    inner_main()


if __name__ == "__main__":
    test_main()
