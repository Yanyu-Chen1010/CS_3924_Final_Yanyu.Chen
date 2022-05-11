# Explore the Stability of Shap Explained Feature Importance
The code written to develop the streamlit interactive tool used for understanding final project of course CS_3924 (Track 2)

My project is inspired by papers: 
> Jabeur, et.al. "Forecasting gold price with the XGBoost algorithm and SHAP interaction values"
> 
> Kumar, et.al. "Problems with Shapley- value-based explanations as feature importance measures"
> 
> Barr, et.al. "Towards Ground Truth Explainability on Tabular Data"
> 
> Renisha Chainani, "FACTORS INFLUENCING GOLD PRICES". (2016)


Jabeur and kumar's works motivate me to test the hypothesis:
> How much does the shap_explained feature attribution change given:
> H1: when interventional correlation between features change
> H2: when fully redundance being added to train data
> H3: when the balance structure of train data change

Folder "sample_data" includes codes help generate clean original data of raw data collected, and folder "WEB_DESIGN" include
