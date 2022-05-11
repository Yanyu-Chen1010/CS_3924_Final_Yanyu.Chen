# Function输入：X_syn, y_label_syn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn import metrics

from synthetic_data.synthetic_data import make_tabular_data
from synthetic_data.utils import tuples_to_cov

# base classifier: xgboost(X), random forest(R), logistic regression(L)
import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import shap
import warnings
warnings.filterwarnings("ignore")


def df_plot_of(syn_matrix, X, Synthetic_by):
    frames = []
    for i in range(0, len(syn_matrix)):
        X_feature, y_label = syn_matrix[i][0], syn_matrix[i][1]
        shap_dict = mean_abs_SHAP_feature_importance_of(X_feature, y_label)

        df_for_plot = pd.DataFrame(shap_dict)
        df_for_plot["features"] = list(X.columns)
        df_for_plot["Synthetic_type"] = Synthetic_by[i]
        frames.append(df_for_plot)

    df_plot = pd.concat(frames, ignore_index=True)

    return df_plot


def df_plot_of_red(syn_matrix, X, Synthetic_by, n_redundant=[0, 4, 6]):
    frames = []
    for i in range(0, len(syn_matrix)):
        X_feature, y_label = syn_matrix[i][0], syn_matrix[i][1]
        shap_dict = mean_abs_SHAP_feature_importance_of(X_feature, y_label)

        df_for_plot = pd.DataFrame(shap_dict)
        df_for_plot["features"] = list(
            X.columns) + ['redundant C='+str(n_redundant[i]) for count in range(1, n_redundant[i]+1)]
        df_for_plot["Synthetic_type"] = Synthetic_by[i]
        frames.append(df_for_plot)

    df_plot = pd.concat(frames, ignore_index=True)

    return df_plot


def mean_abs_SHAP_feature_importance_of(X_feature, y_label):

    total = {}
    model_type_l = ['X', 'R', 'L']
    model_key = ['XGB Classifier',
                 'Random Forest Classifier', 'Logistic Regression']
    clf_lst = [XGBClassifier(eval_metric='auc'), RandomForestClassifier(
        max_depth=4, random_state=0), LogisticRegression(random_state=0)]
    X_train, X_test, y_train, y_test = train_test_split(
        X_feature, y_label, test_size=0.2)

    for clf_idx in [0, 1, 2]:
        model_type = model_type_l[clf_idx]
        clf = clf_lst[clf_idx]
        clf.fit(X_train, y_train)

        # SHAP.Explainer
        if model_type == 'X':
            shap_values = shap.TreeExplainer(clf, X_train).shap_values(
                X_test)      # shap_values shape: (num_input, num_features)
            shap_df = pd.DataFrame(shap_values)
        elif model_type == 'R':
            shap_values = shap.TreeExplainer(clf, X_train).shap_values(
                X_test)      # shap_values shape: (2, num_input, num_features)
            shap_df = pd.DataFrame(shap_values[0])
        elif model_type == 'L':
            shap_values = shap.Explainer(clf, X_train).shap_values(X_test)
            shap_df = pd.DataFrame(shap_values)

        # Calculate mean absolute SHAP values for each feature to get a standard bar plot
        mean_feature_importance = []
        for col in shap_df.columns:
            mean_feature_importance.append(
                np.abs(shap_df[col]-shap_df[col].mean(axis=0)).mean())

        total[model_key[clf_idx]] = mean_feature_importance

    return total  # Output: list of list
