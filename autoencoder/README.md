# AutoEncoder approach

## Structure
```
│
├── configs                  <- Hydra configs
│   ├── data                     <- Model configs
│   ├── model                    <- Model configs
│   ├── path                     <- Project paths configs
│   ├── trainer                  <- W&B configs
│   ├── wandb                    <- W&B configs
│   ├── sweep                    <- Sweep configs
│   │
│   └──config.yaml               <- Main config
│
├── src                      <- Source code
│   ├── data                     <- Data scripts
│   │   ├── datamodule.py             <- Lightning Datamodule scripts
│   │   ├── dataset.py                <- Dataset scripts
│   │   └── utils.py                  <- Data utility scripts
│   │
│   ├── model                    <- Model scripts
│   │   ├── multivae.py               <- Multi-VAE module scripts
│   │   └── recommender.py            <- Lightning Module scripts
│   │
│   ├── trainer.py               <- Trainer scripts
│   └── utils.py                 <- Utility scripts
│
├── main.py                  <- Run training and inference
├── requirements.txt         <- File for installing python dependencies
└── README.md
```

## Setup
```bash
cd autoencoder
conda create -n autoencoder python=3.10 -y
conda activate autoencoder
pip install -r requirements.txt
```

## How to run
```bash
python main.py
```
