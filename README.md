# Medical Cost Regression

## Description

**For an executive results deck, check 'results' folder**

This project builds and implements a regression model in order to bill costs for a individuals using medical services¿, using numerical and categorical information about the clients
- Skills used:
    - Preprocessing pipelines
    - Data visualization
    - Regression Modeling:
        - Linear Regression
        - Support Vector Regression
        - Random Forest
        - Gradient Boosting
    - Model evaluation

The original dataset can be found [here](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Folder Structure
- data
    - raw_data.csv: original dataset
- notebooks
    - EDA.ipynb: Extensive Exploratory Data Analysis notebook using raw data
- results
    - Medical Costs Results.PDF: Executive presentation results deck
    - results_report.ipynb: Results of the models' predictions and model benchmarking. it uses the functions and classes defined in 'src'
    - log.txt: logs files used to keep track of important events (running a model, preprocessing information, etc.)
- src
    - config.py: file used to set global configuration variables
    - utils.py: file used to define global and reusable functions across the ecosystem
    - preprocess.py: file used to define the preprocessing pipeline through classes
    - train.py: file that contanins generic and reusable training functions for binary classification  models
    - evaluate.py: file that contanins generic and reusable testing functions for binary classification  models