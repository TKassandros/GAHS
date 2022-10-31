from pandas.compat import numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.impute import KNNImputer
import weka.core.jvm as jvm
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.core.dataset import create_instances_from_matrices

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import random
from copy import deepcopy
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import datetime
from tqdm import tqdm
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import torch
import time
import json
from sklearn.ensemble import RandomForestRegressor as RF
import warnings

warnings.filterwarnings("ignore")
# Start Weka Engine
jvm.start(max_heap_size="2048m", packages=True)


def foo(l, dtype=int):
    return list(map(dtype, l))


# Transform wind to degrees
def wind_dir(di):
    if di == 'N':
        di = 0
    elif di == 'NNE':
        di = 22.5
    elif di == 'NE':
        di = 45
    elif di == 'ENE':
        di = 67.5
    elif di == 'E':
        di = 90
    elif di == 'ESE':
        di = 112.5
    elif di == 'SE':
        di = 135
    elif di == 'SSE':
        di = 157.5
    elif di == 'S':
        di = 180
    elif di == 'SSW':
        di = 202.5
    elif di == 'SW':
        di = 225
    elif di == 'WSW':
        di = 247.5
    elif di == 'W':
        di = 270
    elif di == 'WNW':
        di = 292.5
    elif di == 'NW':
        di = 315
    elif di == 'NNW':
        di = 337.5
    return di


# Transform degrees to sin
def transform_cyclical(angle,halfperiod):
    return np.sin(angle * np.pi / halfperiod)

# Create more features
def feature_engineering(data, lags, window_sizes):
    # (df, 12, [24, 48])

    data2 = data.copy()
    for col in data.columns:
        for window in window_sizes:
            data2["rolling_mean_" + col + '-' + str(window)] = data[col].rolling(window=window).mean()
            data2["rolling_std_" + col + '-' + str(window)] = data[col].rolling(window=window).std()
            data2["rolling_min_" + col + '-' + str(window)] = data[col].rolling(window=window).min()
            data2["rolling_max_" + col + '-' + str(window)] = data[col].rolling(window=window).max()
            data2["rolling_median_" + col + '-' + str(window)] = data[col].rolling(window=window).median()
            data2["rolling_min_max_diff_" + col + '-' + str(window)] = data2["rolling_max_" + col + '-' + str(window)] - \
                                                                       data2["rolling_min_" + col + '-' + str(window)]
    # create lag features
    cols = data.columns.to_list()

    for col in cols:
        for i in range(1, lags):
            data2[f'{col}_lag_{i}'] = data[col].shift(i)
    return data2


# Feature Selection

def cfs_weka(data_engineered):
    dataset = create_instances_from_matrices(np.array(data_engineered))
    dataset.class_is_last()

    search = ASSearch(classname="weka.attributeSelection.GreedyStepwise")  # , options=["-B"])
    evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
    attsel = AttributeSelection()

    attsel.search(search)
    attsel.evaluator(evaluator)
    attsel.select_attributes(dataset)

    ind_cfs = list(attsel.selected_attributes)[:-1]
    return ind_cfs  # [3, 6, 7 ...]


def RFFI(x, y, number_of_features, estimators):
    RFmodel = RF(n_estimators=estimators, oob_score=True, n_jobs=-1, verbose=1)
    RFmodel.fit(x,
                y.values.ravel())  # anti gia RFmodel.fit(x,y) giati ValueError("sparse multilabel-indicator for y is not supported.")

    ind = RFmodel.feature_importances_
    return ind.argsort()[-number_of_features:]

# Impute missing values and apply feature selection methods
def engineer_select_train(calibration_nodes, ref_pol):  
    calibration_nodes.index = [i for i in range(len(calibration_nodes))]

    x = calibration_nodes.drop([ref_pol], axis=1)
    y = calibration_nodes[ref_pol]

    x = feature_engineering(x, 13, [24, 48])
   
    imputer = sklearn.impute.KNNImputer(n_neighbors=1)  
    dfsx = []
    dfsy = []
    y = y.values.reshape(-1, 1)
    # 
    for i in tqdm(range(0, len(x), 10000)):
        temp = imputer.fit_transform(x[i:i + 10000])
        temp2 = imputer.fit_transform(y[i:i + 10000])
        dfsx.append(temp)
        dfsy.append(temp2)
    x = pd.DataFrame(np.concatenate(dfsx),
                     columns=x.columns)  
    
    y = pd.DataFrame(np.concatenate(dfsy))
    

    ind_cfs = cfs_weka(pd.concat([x, y], axis=1))
    ind_rffi = RFFI(x, y, 35, 50)
    indices = {'CFS': foo(ind_cfs), 'RFFI': ind_rffi.tolist()}
    

    return x, y, indices



def train_batch_models(x, y, indexes, un, df):
    # un: column with stations names
    # create predictions with stations-fold cross-validation for stacking
    all_preds = np.zeros([0, 10])  # 10 (5 models for each subset derived from the two feature selection methods)
    un_num = 1
    for name in tqdm(un):
        print(f'{un_num}. {name}')
        un_num = un_num + 1

        predictions = np.zeros([len(df[df['station'] == name]), 0])

        for key in indexes:
            
            Test_indx = df[df['station'] == name].index
            xtrain = x.iloc[~df.index.isin(Test_indx), :].iloc[:, indexes[key]]
            ytrain = y.iloc[~df.index.isin(Test_indx)]
            xtest = x.iloc[Test_indx, :].iloc[:, indexes[key]]

            XGB100 = XGBRegressor(n_estimators=100,
                                  colsample_bytree=.8,
                                  learning_rate=0.1,
                                  alpha=.1,
                                  tree_method='gpu_hist',
                                  subsample=.8, objective='reg:squarederror',
                                  n_jobs=-1)

            XGB300 = XGBRegressor(n_estimators=300,
                                  colsample_bytree=.8,
                                  learning_rate=0.1,
                                  alpha=.1,
                                  tree_method='gpu_hist',
                                  subsample=.8, objective='reg:squarederror',
                                  n_jobs=-1)

            GBmodel = LGBMRegressor(n_estimators=70, num_leaves=5, feature_fraction=.9, n_jobs=-1, verbose=-1)

            RFmodel = LGBMRegressor(boosting_type="rf",
                                    num_leaves=16,
                                    colsample_bytree=.3,
                                    n_estimators=100,
                                    subsample=.632,  # Standard RF bagging fraction
                                    subsample_freq=1,
                                    n_jobs=-1, verbose=-1)

            LRmodel = LinearRegression(n_jobs=-1)
            


            XGB100.fit(xtrain, ytrain)
            XGB300.fit(xtrain, ytrain)
            RFmodel.fit(xtrain, ytrain)
            GBmodel.fit(xtrain, ytrain)
            LRmodel.fit(xtrain, ytrain)
            

            XGB100_ypred = XGB100.predict(xtest)
            XGB300_ypred = XGB300.predict(xtest)
            RF_ypred = RFmodel.predict(xtest)
            GB_ypred = GBmodel.predict(xtest)
            LR_ypred = LRmodel.predict(xtest)

            



            predictions = np.hstack((predictions,
                                     XGB100_ypred.reshape(-1, 1), XGB300_ypred.reshape(-1, 1), RF_ypred.reshape(-1, 1),
                                     GB_ypred.reshape(-1, 1), LR_ypred.reshape(-1, 1)))

        all_preds = np.vstack((all_preds, predictions))

    pred_names = []
    for key in indexes:
        pred_names = pred_names + [f"XGB100_{key}", f"XGB300_{key}", f"RF_{key}", f"GB_{key}", f"LR_{key}"]

    all_preds = pd.DataFrame(all_preds, columns=pred_names)
    all_preds.index = [i for i in range(all_preds.shape[0])]

    return all_preds


# Read Data
df1 = pd.read_csv('path/PRSA_Data_Aotizhongxin_20130301-20170228.csv')
df2 = pd.read_csv('path/PRSA_Data_Changping_20130301-20170228.csv')
df3 = pd.read_csv('path/PRSA_Data_Dingling_20130301-20170228.csv')
df4 = pd.read_csv('path/PRSA_Data_Dongsi_20130301-20170228.csv')
df5 = pd.read_csv('path/PRSA_Data_Guanyuan_20130301-20170228.csv')
df6 = pd.read_csv('path/PRSA_Data_Gucheng_20130301-20170228.csv')
df7 = pd.read_csv('path/PRSA_Data_Huairou_20130301-20170228.csv')
df8 = pd.read_csv('path/PRSA_Data_Nongzhanguan_20130301-20170228.csv')
df9 = pd.read_csv('path/PRSA_Data_Shunyi_20130301-20170228.csv')
df10 = pd.read_csv('path/PRSA_Data_Tiantan_20130301-20170228.csv')
df11 = pd.read_csv('/=path/PRSA_Data_Wanliu_20130301-20170228.csv')
df12 = pd.read_csv('path/PRSA_Data_Wanshouxigong_20130301-20170228.csv')

df_allStations = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12], ignore_index=True)
un = sorted(df_allStations['station'].unique().tolist())

# Dataframe transformations

# Wind direction
for i, row in df_allStations.iterrows():
    df_allStations.at[i, 'wd'] = wind_dir(df_allStations.wd[i])
    df_allStations.at[i, 'wd'] = transform_cyclical(df_allStations.wd[i],180)


data1 = df_allStations[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']]
target = 'PM2.5'


x, y, indxx = engineer_select_train(data1, target)
all_preds = train_batch_models(x, y, indxx, un, df_allStations)
df_allStations = df_allStations.rename(columns={'station': 'Fold'})
Hybrid_BaseLearners = pd.concat([x, all_preds, y, df_allStations['Fold']], axis=1)
Hybrid_BaseLearners.rename(columns={Hybrid_BaseLearners.columns[-2]: 'PM2.5'},inplace=True)
Hybrid_BaseLearners.to_csv('path/Hybrid_BaseLearners.csv', index=False)
