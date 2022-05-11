
import streamlit as st
import altair as alt

# User input:
# Uploaded_df
# select

# 先写着，看看web_data 能不能run，不能的话就直接单独算出synthetic data的特征
# 然后把图先弄出来

st.title("Explore the Stability of Shap Explained Feature Importance")

selected_model = st.radio(
    "Which Classifier do you want to use?",
    ('XGB Classifier', 'Random Forest Classifier', 'Logistic Regression'))
# st.write(selected_model)
if selected_model in ['XGB Classifier', 'Random Forest Classifier']:
    st.write("{}'s explainer is shap.TreeExplainer".format(selected_model))
elif selected_model in ['Logistic Regression']:
    st.write("{}'s explainer is shap.Explainer".format(selected_model))


# generate the heatmap of Users Dataframe
st.subheader('The Correlation Heatmap of Input original data')


# 然后让 User 选择想创建的 synthetic dataset，即 sub-hypothesis
Synthetic_type = ["Covariance",
                  "Number of fully redundant feature",
                  "Sturcture Balance"]
st.subheader('Synthetic Dataset by {}'.format(Synthetic_type[0]))
# 放图
