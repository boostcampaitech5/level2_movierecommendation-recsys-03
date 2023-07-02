# Sequential approach

## 🧩 Structure
```bash
│
├── configs                    <- Hydra configs
│   ├── data                       <- Data configs
│   ├── model.test                 <- Model pytest configs
│   ├── model                      <- Model configs
│   ├── path                       <- Project paths configs
│   ├── trainer.test               <- Trainer pytest configs
│   ├── trainer                    <- Trainer configs
│   ├── wandb                      <- W&B configs
│   │
│   ├──config.yaml                 <- Main config       
│   └──test.yaml                   <- Pytest config   
│
├── data                       <- Data 
│   └── dummy                      <- Dummydata for test
│
├── src                        <- Source code
│   ├── dataloaders                <- Dataloader scripts
│   │   ├── BERT4RecDataModule.py       <- BERT4Rec dataloader scripts
│   │   ├── S3RecDataModule.py          <- S3Rec dataloader scripts
│   │   ├── SASRecDataModule.py         <- SASRec dataloader scripts
│   │   └── common.py                   <- common dataloader scripts
│   │
│   ├── models                     <- Lightning model scripts
│   │   ├── BERT4Rec.py                 <- BERT4Rec model scripts
│   │   ├── S3Rec.py                    <- S3Rec model scripts
│   │   └── SASRec.py                   <- SASRec model scripts
│   │
│   ├── modules                    <- Modules scripts
│   │   ├── BERT4Rec.py                 <- BERT4Rec module scripts
│   │   ├── S3Rec.py                    <- S3Rec module scripts
│   │   ├── SASRec.py                   <- SASRec module scripts
│   │   └── common.py                   <- common module scripts
│   │
│   ├── config.py                  <- Config dataclass scripts
│   ├── datasets.py                <- Datasets scripts
│   ├── trainer.py                 <- Trainer scripts
│   └── utils.py                   <- Utility scripts
│
├── generate_dummy.ipynb       <- Generate dummy data
├── generate_id2idx_data.ipynb <- Generate indexed data
├── tsv2json.py                <- Convert tsv to json
├── test_main.py               <- Pytest main
├── main.py                    <- Run training and inference
├── requirements.txt           <- File for installing python dependencies
└── README.md
```

## ⚙️ Setup

```bash
cd sequential
conda create -n sequential python=3.10 -y
conda activate sequential
pip install -r requirements.txt
```

## 🚀 How to run
```bash
python main.py
```
