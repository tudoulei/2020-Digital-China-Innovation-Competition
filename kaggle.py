import numpy as np
import pandas as pd
import os
import ast
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import gc

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge

import seaborn as sns
import matplotlib.pyplot as plt 

from glob import glob
import warnings
from dateutil.parser import parse
from tqdm import tqdm

warnings.filterwarnings("ignore")
# pd.options.display.precision = 15
tqdm.pandas() 

tr_path = '/tcdata/hy_round2_train_20200225/'
te_path = '/tcdata/hy_round2_testA_20200225/'
tr_file_paths = os.listdir(tr_path)
te_file_paths = os.listdir(te_path)

df_tr = pd.DataFrame()
for path in tqdm(tr_file_paths):
    df_tr = pd.concat([df_tr, pd.read_csv(tr_path + path, encoding= 'utf-8')],axis=0, ignore_index=True)
    
df_te = pd.DataFrame()
for path in tqdm(te_file_paths):
    df_te = pd.concat([df_te, pd.read_csv(te_path + path, encoding= 'utf-8')],axis=0, ignore_index=True)
    
df = pd.concat([df_tr, df_te], axis = 0, ignore_index=True)

df['time'] = df['time'].progress_apply(lambda x: parse(x[:2] + '-' + x[2:]))
df['type'] = df['type'].map({'拖网':0,'围网':1,'刺网':2})
df['type'] = df['type'].fillna(-1)
df         = df.sort_values(['渔船ID','time'])
df.to_csv('/data/df.csv',index = None)

df['time_next']    = df.groupby(['渔船ID'])['time'].shift(-1)
df['time_last']    = df.groupby(['渔船ID'])['time'].shift(1)

df['time_to_next'] = df['time_next'] - df['time']
df['time_to_last'] = df['time'] - df['time_last']

df['time_to_next'] = df['time_to_next'].dt.total_seconds()
df['time_to_last']      = df['time_to_last'].dt.total_seconds()
df['kilo_meter_approx'] = df['速度'].values * df['time_to_next'].map(lambda x: x/3600).values

df['speed_diff'] = df.groupby(['渔船ID'])['速度'].shift(1)
df['speed_diff'] = df['速度'].values - df['speed_diff'].values

df['direction_diff'] = df.groupby(['渔船ID'])['方向'].shift(1)
df['direction_diff'] = df['方向'].values - df['direction_diff'].values

df['x_diff'] = df.groupby(['渔船ID'])['x'].shift(1)
df['x_diff'] = df['x'].values - df['x_diff'].values
df['y_diff'] = df.groupby(['渔船ID'])['y'].shift(1)
df['y_diff'] = df['y'].values - df['y_diff'].values

def get_mode(x, i):
    t = x.value_counts()
    try:
        return t.index[i]
    except:
        return -1  

def get_time_features(df, main_key = '渔船ID'):
    df_time           = pd.DataFrame()
    df_time[main_key] = df[main_key].unique()
    
    df['hour_minutes']    = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['hour']            = df['time'].dt.hour  
    df_grp = df.groupby([main_key])
    
    ################### 时间的众数 ##############
    dic1 = df_grp['hour'].apply(lambda x:  get_mode(x,0)).to_dict()
    dic2 = df_grp['hour'].apply(lambda x:  get_mode(x,1)).to_dict()
    df_time['time_mode_0'] = df_time[main_key].map(dic1).values
    df_time['time_mode_1'] = df_time[main_key].map(dic2).values 
    
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp['hour_minutes'].agg(opt).to_dict()
        col_name = main_key + '_hour_minutes_' + opt
        df_time[col_name] = df_time[main_key].map(dic).values
    
    ################### 时间差的分布 ##############
    for opt in ['mean','std','quantile','skew','median', 'max', 'min']: 
        dic = df_grp['time_to_next'].agg(opt).to_dict()
        col_name = main_key + '_time_to_next_' + opt
        df_time[col_name] = df_time[main_key].map(dic).values
   
    return df_time


df_time = get_time_features(df, main_key = '渔船ID')



def get_zero_num(x):
    return np.sum(x == 0) 

def get_less_zero_num(x):
    return np.sum(x < 0) 

def get_big_zero_num(x):
    return np.sum(x > 0) 

def get_speed_features(df, main_key = '渔船ID', fea_col = '速度'):
    df_speed           = pd.DataFrame()
    df_speed[main_key] = df[main_key].unique()
    df_grp = df.groupby([main_key]) 
    
    df_tmp  = df.loc[df[fea_col] > 0].copy()
    df_grp1 = df_tmp.groupby([main_key]) 
    
    dic      = df_grp[fea_col].apply(lambda x: get_zero_num(x)).to_dict()
    col_name = main_key + '_{}_zero_num'.format(fea_col)
    df_speed[col_name] = df_speed[main_key].map(dic).values 
    
    ################### 速度的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_speed[col_name] = df_speed[main_key].map(dic).values 
    df_speed[main_key + '_{}_'.format(fea_col) + 'gap'] = df_speed[main_key + '_{}_'.format(fea_col) + 'max'] - df_speed[main_key + '_{}_'.format(fea_col) + 'min']
    for q in [0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95]:
        dic = df_grp[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_quantile_{}'.format(fea_col,q) 
        df_speed[col_name] = df_speed[main_key].map(dic).values
    
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic      = df_grp1[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_big_zero_'.format(fea_col) + opt
        df_speed[col_name] = df_speed[main_key].map(dic).values
    
    df_speed[main_key + '_{}_big_zero_'.format(fea_col) + 'gap'] = df_speed[main_key + '_{}_big_zero_'.format(fea_col) + 'max'] - df_speed[main_key + '_{}_big_zero_'.format(fea_col) + 'min']
  
    for q in [0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95]:
        dic = df_grp1[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_big_zero_quantile_{}'.format(fea_col,q) 
        df_speed[col_name] = df_speed[main_key].map(dic).values
    
    ################### 速度差的分布 ##############
    for opt in ['mean','std','skew','median', 'max', 'min']: 
        dic = df_grp['speed_diff'].agg(opt).to_dict()
        col_name = main_key + '_speed_diff_' + opt
        df_speed[col_name] = df_speed[main_key].map(dic).values
        
    dic      = df_grp1['speed_diff'].apply(lambda x: get_zero_num(x)).to_dict()
    col_name = main_key + '_speed_diff_zero_num'
    df_speed[col_name] = df_speed[main_key].map(dic).values
    
    dic      = df_grp1['speed_diff'].apply(lambda x: get_big_zero_num(x)).to_dict()
    col_name = main_key + '_speed_diff_big_zero_num'
    df_speed[col_name] = df_speed[main_key].map(dic).values
    
    dic      = df_grp1['speed_diff'].apply(lambda x: get_less_zero_num(x)).to_dict()
    col_name = main_key + '_speed_diff_less_zero_num'
    df_speed[col_name] = df_speed[main_key].map(dic).values
     
    return df_speed


df_speed = get_speed_features(df, main_key = '渔船ID', fea_col = '速度')


def get_meter_features(df, main_key = '渔船ID', fea_col = 'kilo_meter_approx'):
    df_meter           = pd.DataFrame()
    df_meter[main_key] = df[main_key].unique()
    df_grp = df.groupby([main_key]) 
    
    df_tmp  = df.loc[df[fea_col] > 0].copy()
    df_grp1 = df.groupby([main_key]) 
    
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min','sum']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_meter[col_name] = df_meter[main_key].map(dic).values
     
    return df_meter


df_meter = get_meter_features(df, main_key = '渔船ID', fea_col = 'kilo_meter_approx')



def get_direction_features(df, main_key = '渔船ID', fea_col = '方向'):
    df_direction           = pd.DataFrame()
    df_direction[main_key] = df[main_key].unique()
    df_grp = df.groupby([main_key]) 
    
    df_tmp  = df.loc[df[fea_col] > 0].copy()
    df_grp1 = df.groupby([main_key]) 
    
    dic      = df_grp[fea_col].apply(lambda x: get_zero_num(x)).to_dict()
    col_name = main_key + '_{}_zero_num'.format(fea_col)
    df_direction[col_name] = df_direction[main_key].map(dic).values
    
    dic      = df_grp['direction_diff'].apply(lambda x: get_zero_num(x)).to_dict()
    col_name = main_key + '_direction_diff_zero_num'
    df_direction[col_name] = df_direction[main_key].map(dic).values
    
    dic      = df_grp[fea_col].apply(lambda x: get_less_zero_num(x)).to_dict()
    col_name = main_key + '_direction_diff_less_zero_num'
    df_direction[col_name] = df_direction[main_key].map(dic).values
    
    dic      = df_grp[fea_col].apply(lambda x: get_big_zero_num(x)).to_dict()
    col_name = main_key + '_direction_diff_big_zero_num'
    df_direction[col_name] = df_direction[main_key].map(dic).values
     
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_direction[col_name] = df_direction[main_key].map(dic).values
    
    
    for q in [0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95]:
        dic      = df_grp[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_quantile_{}'.format(fea_col,q) 
        df_speed[col_name] = df_speed[main_key].map(dic).values 
    
    
    ################### 时间的分布 ############## 
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp1[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_big_zero'.format(fea_col) + opt
        df_direction[col_name] = df_direction[main_key].map(dic).values 
    
    for q in [0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95]:
        dic      = df_grp1[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_big_zero_quantile_{}'.format(fea_col,q) 
        df_speed[col_name] = df_speed[main_key].map(dic).values
    
    ################### 速度差的分布 ##############
    for opt in ['mean','std','skew','median', 'max', 'min']: 
        dic = df_grp['direction_diff'].agg(opt).to_dict()
        col_name = main_key + '_direction_diff_' + opt
        df_direction[col_name] = df_direction[main_key].map(dic).values 
    return df_direction


df_direction = get_direction_features(df, main_key = '渔船ID', fea_col = '方向')


def get_xy_features(df, main_key = '渔船ID'):
    df_xy           = pd.DataFrame()
    df_xy[main_key] = df[main_key].unique()
    df_grp = df.groupby([main_key])  
    
    fea_col = 'x'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values 
    
    fea_col = 'x_diff'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values
    
    
    fea_col = 'y'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values
    
    
    fea_col = 'y_diff'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values
        
    fea_col = 'x_minus_y'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values 
    
    for q in [0.025,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.975]:
        dic = df_grp[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_quantile_{}'.format(fea_col,q)  
        df_xy[col_name] = df_xy[main_key].map(dic).values 
    
    fea_col = 'x_div_y'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values 
         
    for q in [0.025,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.975]:
        dic = df_grp[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_quantile_{}'.format(fea_col,q)  
        df_xy[col_name] = df_xy[main_key].map(dic).values 
         
    fea_col = 'x_2_y_2'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values 
         
    for q in [0.025,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.975]:
        dic = df_grp[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_quantile_{}'.format(fea_col,q)  
        df_xy[col_name] = df_xy[main_key].map(dic).values
    
    fea_col = 'x_minus_mean'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values 
         
    for q in [0.025,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.975]:
        dic = df_grp[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_quantile_{}'.format(fea_col,q)  
        df_xy[col_name] = df_xy[main_key].map(dic).values 
    
    fea_col = 'y_minus_mean'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values 
         
    for q in [0.025,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.975]:
        dic = df_grp[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_quantile_{}'.format(fea_col,q)  
        df_xy[col_name] = df_xy[main_key].map(dic).values  
        
    fea_col = 'x_minus_mean_div_std'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values 
         
    for q in [0.025,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.975]:
        dic = df_grp[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_quantile_{}'.format(fea_col,q)  
        df_xy[col_name] = df_xy[main_key].map(dic).values 
    
    fea_col = 'y_minus_mean_div_std'
    ################### 时间的分布 ##############
    for opt in ['mean','quantile','skew','median','max', 'min']:
        dic = df_grp[fea_col].agg(opt).to_dict()
        col_name = main_key + '_{}_'.format(fea_col) + opt
        df_xy[col_name] = df_xy[main_key].map(dic).values 
         
    for q in [0.025,0.05,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.95,0.975]:
        dic = df_grp[fea_col].quantile(q).to_dict()
        col_name = main_key + '{}_quantile_{}'.format(fea_col,q)  
        df_xy[col_name] = df_xy[main_key].map(dic).values  
        
    return df_xy  

df['x_minus_y']    = df['x'].values - df['y'].values
df['x_div_y']      = df['x'].values / df['y'].values
df['x_2_y_2']      = df['x'].values ** 2 + df['y'].values** 2 
df['x_minus_mean'] = df['x'].values - df.groupby('渔船ID')['x'].transform('mean').values
df['y_minus_mean'] = df['y'].values - df.groupby('渔船ID')['y'].transform('mean').values
df['x_minus_mean_div_std'] = (df['x'].values - df.groupby('渔船ID')['x'].transform('mean').values) / df.groupby('渔船ID')['x'].transform('std').values
df['y_minus_mean_div_std'] = (df['y'].values - df.groupby('渔船ID')['y'].transform('mean').values) / df.groupby('渔船ID')['x'].transform('std').values
df['x_2_y_2']      = df['x'].values ** 2 + df['y'].values** 2


df_xy = get_xy_features(df, main_key = '渔船ID')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

eval_fun = f1_score
def lgb_model_validation(df,  features, cate_vars = None, test_df = None ):  

    params = {
    'num_leaves': 2 ** 10,
    'learning_rate': 0.005,
    'min_child_samples': 20,
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 20000,
    'metric': 'multi_logloss',
    'num_class': 3,
    'feature_fraction': .75,
    'bagging_fraction': .75,
    'seed': 99,
    'num_threads': 8,
    'verbose': -1
    }
    
    MAX_ROUNDS = 20000
    val_pred = []
    test_pred = []
    feature_importance = None
    models = [] 
    
    kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=123)     
    preds_train = np.zeros((len(df), 3), dtype = np.float) 
    train_loss = []; test_loss = []
    
    X_train = df[features].copy()
    y_train = df['type'].values
    scores  = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        train_df,train_label = X_train.iloc[tr_idx, :].copy() ,y_train[tr_idx]
        val_df,val_label     = X_train.iloc[val_idx, :].copy() ,y_train[val_idx]
         
        if cate_vars is not None:
            dtrain = lgb.Dataset(train_df, label = train_label, categorical_feature=cate_vars)  
            dval   = lgb.Dataset(val_df,   label=val_label, reference=dtrain) 
        else:
            dtrain = lgb.Dataset(train_df, label = train_label)  
            dval   = lgb.Dataset(val_df,   label=val_label, reference=dtrain) 

        bst = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,valid_sets=[dtrain, dval],  early_stopping_rounds=300, verbose_eval=50)
 

        train_loss.append(eval_fun(y_true=train_label, y_pred=np.argmax(bst.predict(train_df), 1), average='macro'))
        test_loss.append(eval_fun(y_true=val_label, y_pred=np.argmax(bst.predict(val_df), 1),  average='macro'))

        preds_train[val_idx] = bst.predict(val_df)[:] 
        
        print('{0}: Train {1:0.7f} Val {2:0.7f}/{3:0.7f}'.format(fold, train_loss[-1], test_loss[-1], np.mean(test_loss)))
        print('-' * 50)
        
        models.append(bst)
        f_importance = pd.DataFrame()
        f_importance['fea'] = features
        f_importance['imp'] = bst.feature_importance("gain")
        f_importance['fold'] = fold
        
        if feature_importance is None:
            feature_importance = f_importance
        else:
            feature_importance = pd.concat([feature_importance, f_importance],axis=0,ignore_index=True) 
            
        if test_df is not None:
            test_pred.append(bst.predict(
                test_df[features], num_iteration=bst.best_iteration or MAX_ROUNDS))
    if test_df is None:
        return models, feature_importance, test_loss, preds_train, None
    else:
        return models, feature_importance, test_loss, preds_train, test_pred

df_fea_label = df_label.merge(df_time,          on =['渔船ID'], how='left')
df_fea_label = df_fea_label.merge(df_speed,     on =['渔船ID'], how='left')
df_fea_label = df_fea_label.merge(df_meter,     on =['渔船ID'], how='left')
df_fea_label = df_fea_label.merge(df_direction, on =['渔船ID'], how='left')
df_fea_label = df_fea_label.merge(df_xy,        on =['渔船ID'], how='left')
fea_cols = [col for col in df_fea_label.columns if col not in ['渔船ID','type']]  
models, feature_importance, test_loss, preds_train, test_pred = lgb_model_validation(df = df_fea_label, features=fea_cols, test_df=None)


test_pred = np.mean(test_pred,axis=0) 
df_test          = df_fea_label.loc[df_fea_label.type < 0].copy()
df_test['label'] = np.argmax(test_pred, 1)  
df_test['label'] = df_test['label'].map({0:'拖网',1:'围网',2:'刺网'})
df_test[['渔船ID', 'label']].to_csv('baseline.csv',index=None, header=None)
