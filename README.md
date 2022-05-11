# Explore the Stability of Shap Explained Feature Importance
The code written to develop the streamlit interactive tool used for understanding final project of course CS_3924 (Track 2)

<br/>

My project is inspired by works: 
> Jabeur, et.al. "Forecasting gold price with the XGBoost algorithm and SHAP interaction values"
> 
> Kumar, et.al. "Problems with Shapley- value-based explanations as feature importance measures"
> 
> Barr, et.al. "Towards Ground Truth Explainability on Tabular Data" [CapitalOne-Github: "Create Synthetic Data by correlation matrix"](https://github.com/capitalone/synthetic-data)
> 
> Renisha Chainani, "FACTORS INFLUENCING GOLD PRICES". (2016)
> 
> [Correlation Matrix Heatmap Plot](https://seaborn.pydata.org/examples/many_pairwise_correlations.html)

<br/>

Jabeur and kumar's works motivate me to test the hypothesis:
> How much does the shap_explained feature attribution change given:
> 
> H1: when interventional correlation between features change
> 
> H2: when fully redundance being added to train data
> 
> H3: when the balance structure of train data change

<br/>

Folder "sample_data" includes codes help generate clean original data of raw data collected, and folder "Interactive_tool" include files used to mathematically processs the orginal data to obatin different kind of synthetic data. The file "web_data_upload.py" in folder "Interactive_tool" will give us the Streamlit Interactive Website, where users can input cleaned orginal data to test how the SHAP explained feature importance value change with different synthetic data of various data structure and characteristics. 

<br/>

The source of collecting raw features data to generate "Original Data of Gold Price" in folder sample data are:
> [Gold Price](https://www.gold.org/)
> 
> [Metal Commodities' Price and Dollar Index](https://www.investing.com/)
> 
> [Geopolotical and Economic Policy Uncertainty indicators](https://www.policyuncertainty.com/gpr.html)

<br/>

The source of collecting raw features data to generate "Original Data of census income" in folder sample data is:
> [UCI Census-Income (KDD) Data Set](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD))
