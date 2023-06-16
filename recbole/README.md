# RecBole

## Structure - TODO
```
│
├── configs                   <- Hydra configs
│   └── data.yaml                <- Data configs
│
├── src                       <- Source code
│   ├── models                   <- Graph model scripts
│   └── utils.py                 <- Utility scripts
│
├── main.py                  <- Run training and inference
├── requirements.txt         <- File for installing python dependencies
└── README.md
```


## Setup
```bash
cd ~/level2_movierecommendation-recsys-03/recbole
conda init
(base) . ~/.bashrc
(base) conda create -n rec python=3.9 -y
(base) conda activate rec
(recbole) pip install -r requirements.txt
```

## How to run
```bash
python main.py
```