# Medical Cost Regression

## Description

**For an executive results deck, check 'results' folder or click [here](https://drive.google.com/file/d/1aQCh0yVUKIF1y0PdP9GNjwmavwqsRq6_/view?usp=sharing)**

This project builds and implements a regression model to predict bill costs for a individuals using medical services. It uses numerical and categorical information about the clients to build the model
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
