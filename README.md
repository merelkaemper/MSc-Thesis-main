## NS Data & Models
**Repository structure:** <br/>

* **data/** -- data folder to add NS archive train data from (https://www.rijdendetreinen.nl/open-data/treinarchief)
* **stations_data/** -- data used for explaring additional dataset (see Exploratory_Data_Analysis.ipynb)
* **features/** -- data folder where calculated features file will be based after running 0_Feature_Engineering.py
* **results_xgb/** -- data folder to put model output from baseline model where the XGBoost Classifier is used. (running 1_classification)
* **results_classifiers/** -- data folder to put model output (running 1b_classification)
* **notebooks/** -- notebooks for visulizations (e.g., data preparation, eda and final figures)
* **src/** -- other supporting code for feature extracting


**Steps:** <br/>

* **env.txt** -- create conda environment (```$ conda create --name <env> --file <this file> ```)
* **Data_Preparation.ipynb** -- prepare dataset to extract features
* **0_Feature_Engineering.py** -- building feature matrix for the model 
* **1_Classification.py** -- running baseline model on NS data
* **1b_Classification.py** -- running new models on NS data for comparison

