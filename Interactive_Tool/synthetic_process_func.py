import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn import metrics

from synthetic_data.synthetic_data import make_tabular_data
from synthetic_data.utils import tuples_to_cov

'''
Self-written Functions:
    plot_corr_heatmap(df_features)
    tup_lst_of(X_corr)
    col_map_of(X)
    change_balance(X_syn, y_label_syn, balance=[])
    New_corr(X, base_feature, dependent)
    get_near_psd(A)
    is_pos_semidef(x)
    obtain_X_y_syn()
'''


def plot_corr_heatmap(df_features):
    corr = df_features.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(6, 4))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return plt.gcf().set_size_inches(10, 8)


# 自定义 cov_matrix 的特征
def tup_lst_of(X_corr):
    num_iter = len(X_corr)
    tup_lst_corr = []   # tuple = (sym_i, sym_j, correlation)
    for i in range(0, num_iter):
        for j in range(0, num_iter):
            tup = ('x'+str(i), 'x'+str(j), X_corr[i, j])
            tup_lst_corr.append(tup)
    return tup_lst_corr


# 对user输入的 tabular-data columns命名与标注index
def col_map_of(X):
    val = np.arange(0, len(X.columns), 1)
    key = ["x"+str(num) for num in val]
    col_map = {}
    for i in val:
        col_map[key[i]] = val[i]
    return col_map


def change_balance(X_syn, y_label_syn, balance):
    # balance = [0.5, 0.5]
    class_1_count = np.count_nonzero(y_label_syn)
    class_0_count = len(y_label_syn)-class_1_count
    count_per_class = [class_0_count, class_1_count]
    # print(count_per_class)

    idx_class_1 = list(np.nonzero(y_label_syn)[0])
    idx_class_0 = np.where(y_label_syn == 0)[0]
    idx_class = [idx_class_0, idx_class_1]

    if class_1_count > class_0_count:
        base_label = [1, 0]
    elif class_1_count <= class_0_count:
        base_label = [0, 1]  # base = class_0, whose count is more

    # print(type(balance))
    if balance[0] != 0.5:
        base_idx = base_label[0]
        total = int(np.round(count_per_class[base_idx]/balance[1], 0))
    # case: fully balance
    elif balance[0] == 0.5:
        total = int(np.round(count_per_class[base_label[1]]/balance[1], 0))

    num_1 = int(np.round(total*balance[base_label[0]], 0))
    num_2 = int(np.round(total*balance[base_label[1]], 0))

    front = idx_class[base_label[0]][:num_1]
    back = idx_class[base_label[1]][:num_2]

    front_X = [X_syn[idx, :] for idx in front]
    back_X = [X_syn[idx, :] for idx in back]

    front_y = [y_label_syn[idx] for idx in front]
    back_y = [y_label_syn[idx] for idx in back]

    X_bal = pd.DataFrame(np.concatenate((front_X, back_X)))
    y_bal = pd.DataFrame(np.array(front_y+back_y))

    return X_bal, y_bal


# 要先创建 New X_corr
def New_corr(X, base_feature, dependent):
    corr = X.corr().to_numpy()
    for dep_col in dependent:
        user_i = X.columns.get_loc(base_feature)
        user_j = X.columns.get_loc(dep_col)

        for i in range(0, len(X)):
            for j in range(0, len(X.columns)):
                if (i, j) == (user_i, user_j) and i != j:
                    corr[i, j] = corr[j, i] = 0.1
    # print(corr)
    return corr  # output: np.array


# Ref: https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix/63131250#63131250
def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)


# OBTAIN SYNTHETIC DATA
def obtain_X_y_syn(
        X,
        n_samples,
        base_feature=None,
        independent=None,
        n_redundant=0,
        balance=[0.5, 0.5]):

    if (base_feature is None) and (independent is None):
        X_corr = X.corr().to_numpy()
    else:
        X_corr = New_corr(X, base_feature, independent)

    col_map = col_map_of(X)
    cov = tuples_to_cov(tup_lst_of(X_corr), col_map)

    if not is_pos_semidef(cov):
        # Ref: https://www.includehelp.com/python/creating-symmetric-matrices.aspx
        temp = np.matmul(cov, cov.T)
        cov = np.round(get_near_psd(temp), 3)

    # print(is_pos_semidef(cov))
    X_syn, y_syn = make_tabular_data(n_samples=n_samples,
                                     n_informative=len(col_map),
                                     n_redundant=n_redundant,
                                     cov=cov, col_map=col_map, seed=2022)
    X_bal, y_bal = change_balance(X_syn, y_syn, balance=balance)
    return [X_bal, y_bal]
