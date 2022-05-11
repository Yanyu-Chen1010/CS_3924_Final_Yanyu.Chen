
from PIL import Image
import streamlit as st
import altair as alt
import pandas as pd

from shap_mean_importance_value import mean_abs_SHAP_feature_importance_of
from shap_mean_importance_value import df_plot_of
from shap_mean_importance_value import df_plot_of_red
from altair_compare_plot import plot_comparison
import synthetic_process_func as syn


st.title("Explore the Stability of Shap Explained Feature Importance")
st.markdown("**Author: Yanyu, Chen (Track 2)**")

st.subheader("Literature Review")
st.image(Image.open('Jabeur.png'), use_column_width='always')
st.markdown('''

> Jabeur, et.al. "Forecasting gold price with the XGBoost algorithm and SHAP interaction values"
>
> Kumar, et.al. "Problems with Shapley- value-based explanations as feature importance measures"
>
> Brian, Barr et.al. "Towards ground truth explainability on tabular data"

''')
st.image(Image.open('kumar paper.png'), use_column_width='always')


st.subheader("Hypothesis to verify")
st.markdown('''How much does the shap_explained feature attribution change,
>> H1: when interventional correlation between features change
>>
>> H2: when fully redundance being added to train data
>>
>> H3: when the balance structure of train data change''')


st.subheader("Methodology")
st.markdown('''
The CSV file uploaded by user will be transformed into dataframe 
for later synthetic data creation. The dataframe (df_original) will be parsed into

> X: df_features
>
> y: df_label

The model examined in this tool are all binary-classification, 
including: 

> XGBoost.Classifier, Random Forest, and Logistic Regression.

User can later test the Shap's explanabily of the selected model.''')

st.markdown("**1. Interventional Redundance**")
st.latex(r'''X_{3} =X_{1}\cdot X_{2}''')
st.markdown(''' Above equation is quoted from "Kumar,section 3.1.2", where X3 is the added redundant feature when we generate 
synthetic datasets based upon original input data. 
X1, X2 are features originally given in original input data (X)''')
st.latex(r'''X_{base} =X_{indep-1}\cdot X_{indep-2} ''')
st.markdown('''
In out tool, if user choose create synthetic data by change covariance,
that means we will add the redundant X-base to X, then use "make-tabular-data" to generate synthetic dataset.
''')


st.markdown("**2. Fully Redundance**")
st.latex(
    r'''X_{synthetic} = X + C = X + n_{redundant}\cdot \text{columns from X}''')
st.markdown(''' 
In referenc to "Kumar, section 3.1.1", A and B are feature set originally in X. C is the added redundance.
Unlike interventioal redundance, C here is the direct copy of B. 

In this project, we assume X = A+B, C = n_redundant copied columns from X. 
n_redundant decides how many columns that C copied from columns of X.''')


st.subheader('''Please upload the CSV file data you want to probe.''')
st.markdown(''' The uploaded table must have y_label in the last columns, and the columns number must larger than 6''')
# Upload files 限制 X.columns_number > 6（因为redundant 不能为负, 且不能超过 X.columns number的长度）
# 且format必须是label在最后一个column
uploaded_file = st.file_uploader("Choose a file")
X, y = 0, 0
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X = df.loc[:, df.columns != df.columns[-1]]  # features table
    y = df.loc[:, df.columns == df.columns[-1]]  # label column
    st.write(df.head())
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown(
        "**The feature correlation matrix of input original data is described below:**")
    st.pyplot(syn.plot_corr_heatmap(X))  # Print X_heatamap_Correlation


syn_type_all = []
selected_model = 0
with st.sidebar:
    st.subheader('''DO NOT select before upload CSV file''')
    syn_type_all = st.multiselect(
        'Please select the synthetic dataset creation method:',
        ('Change covariance', 'Change number of fully redundant feature', 'Change Structure Balance of tabular data'))
    selected_model = st.selectbox(
        "Please select the Classifier model to explain:",
        ('XGB Classifier', 'Random Forest Classifier', 'Logistic Regression'))


# 生成 tabular data: 所有的synthetic data,都是靠original data建立
st.subheader("Set initial parameters")
st.markdown(
    "**I**. Please select the **number of samples** to generate in synthetic dataset")
n_samples = st.number_input(
    '* n_samples must in interval [100, 1000]', 100, 1000)
st.markdown(
    '''
    **II**. The choice of **n_redundance_fully** is fixed to: 
    >> n_redundance=[0, 4, 6]
    ''')
st.markdown('''
    **III**. The choice of **balance structure** is fixed to: 
    > Balance structure = (Proportion of Class 0, Proportion of Class 1)
    >
    >> Balanced=(0.5, 0.5) 
    >> 
    >> Slightly Imbalance=(0.4, 0.6)  
    >>
    >> Very Imbalance=(0.1, 0.9)''')


cov_syn = []
red_syn = []
imb_syn = []
enter = False
for syn_type in syn_type_all:
    if syn_type == 'Change covariance':
        st.subheader(
            "You selected synthetic method: {}".format(syn_type))
        st.markdown(
            '''
            Please select the base feature(X3) and the two independent feature (X1, X2) you want to probe.
            X1, X2 should have strong correlation with X3. This tool will shrink their strong correlation to near zero in order to verify how the 
            shap_explained feature importance attribution change. 
            ''')
        st.markdown(
            '''
            In general, creating synthetic dataset by change covariance, is changing interventional correlation between features.
            ''')
        st.markdown(
            '''
            (Please use the heatmap describing original data above to help you make selection.)
            ''')
        # Output: cov_syn = [[X_syn_0, y_syn_0], [X_syn_1, y_syn_1]]
        base_feature = st.selectbox(
            "Please select one base feature:", (X.columns))
        independent = st.multiselect(
            "Please select only two independent features:", (X.columns))
        if len(independent) == 2:
            enter = True
        cov_syn.append(syn.obtain_X_y_syn(X, n_samples))
        cov_syn.append(syn.obtain_X_y_syn(
            X, n_samples, base_feature, independent))

    elif syn_type == 'Change number of fully redundant feature':
        n_redundant_lst = [0, 4, 6]
        # Output: redundant_syn = [ [X_r0, y_r0], [X_r4, y_r4], [X_r6, y_r6] ]
        L = len(n_redundant_lst)
        for i in range(0, L):
            n_redundant = n_redundant_lst[i]
            temp_lst = syn.obtain_X_y_syn(
                X, n_samples, n_redundant=n_redundant)
            red_syn.append(temp_lst)

    elif syn_type == 'Change Structure Balance of tabular data':
        balance_lst = [[0.5, 0.5], [0.4, 0.6], [0.1, 0.9]]
        # Output: imbalance_syn = [ [X_b55, y_b55], [X_b46, y_46], [X_b19, y_b19] ]
        L = len(balance_lst)
        for i in range(0, L):
            balance = balance_lst[i]
            temp_lst = syn.obtain_X_y_syn(
                X, n_samples, balance=balance)
            imb_syn.append(temp_lst)


# 上面 for-loop 走完，3 group的list也会填满
# 下一步：
# 计算each group 对应的 df_plot ———— 转换 synthetic data into "df_for_plot"
Synthetic_bycov = ['Original Covariance',
                   'interventional less-correlated Covariance']
n_redundant_lst = [0, 4, 6]
Synthetic_byredundant = ['Causal Redundant C = ' +
                         str(count) for count in n_redundant_lst]
Synthetic_byimbalance = ['Balanced', 'Slightly Imbalance', 'Very Imbalance']

scale_color_cov = alt.Scale(domain=Synthetic_bycov,
                            range=['#aec7e8', '#9467bd'])
scale_color_red = alt.Scale(domain=Synthetic_byredundant, range=[
                            '#aec7e8', '#9467bd', '#1f77b4'])
scale_color_imb = alt.Scale(domain=Synthetic_byimbalance, range=[
                            '#aec7e8', '#9467bd', '#1f77b4'])


st.subheader("Results: Performance of different synthetic datasets")
st.markdown('''This area will be blank if no original CSV data being uploaded, or no synthetic method being selected''')
if enter == True:
    with st.container():
        if cov_syn != []:
            df_plot_cov = df_plot_of(cov_syn, X, Synthetic_bycov)
            # 最后一步：
            # Altair -- plot_comparison(df_plot, selected_model, scale_color, title)
            chart1 = plot_comparison(
                df_plot_cov, selected_model, scale_color_cov)
            st.subheader(
                "H1. Change Covariance (Add interventional reduandance)")
            st.altair_chart(chart1)

        if red_syn != []:
            # redundant 的 df_plot func 不太一样
            df_plot_red = df_plot_of_red(red_syn, X, Synthetic_byredundant)
            chart2 = plot_comparison(
                df_plot_red, selected_model, scale_color_red)
            st.subheader("H2. Add fully redundance")
            st.altair_chart(chart2)

        if imb_syn != []:
            df_plot_imb = df_plot_of(imb_syn, X, Synthetic_byimbalance)
            chart3 = plot_comparison(
                df_plot_imb, selected_model, scale_color_imb)
            st.subheader("H3. Change structure balance")
            st.altair_chart(chart3)
