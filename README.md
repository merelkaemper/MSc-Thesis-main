## US Air Data & Models
Much of this code is from the article by Lei et al. Forecasting the evolution of fast-changing transportation networks using machine learning (2022) <br/>
Original GitHub: https://github.com/amarallab/transportation_network_evolution/blob/master/README.md?plain=1
Their code is implemented as a baseline for this Thesis and eventually improved to fit NS Train data (see other breach)

**Repository structure:** <br/>

* **data/** -- 
    * **raw_usair_data/** -- data folder to put raw US air transportation data (download at https://doi.org/10.21985/n2-9r77-p344)
    * **features/** -- data folder to put calculated topological features (running raw2features)
* **results/** -- data folder to put model output (running classification and longterm_prediction)
* **notebooks/** -- notebooks for visualizations 
* **src/** -- other supporting codes for analysis and visualization


**Steps:** <br/>

* **env.txt** -- create conda environment (```$ conda create --name <env> --file <this file> ```)
* **0_raw2features_usair.py** -- building feature matrix for the models
* **1_classification.py** -- running different models for results 
