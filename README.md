# GPS-Mamba
Official Implementation of "GPS-Mamba: Graph Permutation Scanning State Space Model for Multivariate Time Series Forecasting"

## Preparation
All the permutation results of datasets can be obtained from [here](https://github.com/YunMeiGongRen/GPS-Mamba/releases/download/Permutation1.0/database.zip). 

Please download database.zip, then unzip and copy it to the folder `./database` in our repository.

## Running the Code
### Installation
The full libraries list is provided as a `requirements.txt` in this repo. Please create a virtual environment with `conda` or `venv` and run
```
pip install -r requirements.txt
``` 
### Train and evaluate
```
# ECL
bash ./scripts/multivariate_forecasting/ECL.sh

# Traffic
bash ./scripts/multivariate_forecasting/Traffic.sh

# Weather
bash ./scripts/multivariate_forecasting/weather.sh

# Solar-Energy
bash ./scripts/multivariate_forecasting/Solar.sh

# SD
bash ./scripts/multivariate_forecasting/sd.sh

# PEMS
bash ./scripts/multivariate_forecasting/PEMS03.sh
bash ./scripts/multivariate_forecasting/PEMS08.sh
```

## Laplace Scanning Algorithm
[Laplace_Scanning.ipynb](https://github.com/YunMeiGongRen/GPS-Mamba/blob/main/Laplace_Scanning.ipynb) provides an example code of reproducing Laplace Scanning Algorithm. You can modify the content according to your dataset requirements.
