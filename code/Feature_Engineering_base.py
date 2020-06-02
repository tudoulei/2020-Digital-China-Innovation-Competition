import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
from config import config
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')

type_map_rev = {0: '拖网', 1: '围网', 2: '刺网'}

from config import config
config = config()

feature_path = "./temp/input/"
basefeature_save_path = "./temp/basefea/"
os.makedirs(basefeature_save_path,exist_ok=1)

def group_feature(df, key, target, aggs):   
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t

def cc(x):
    return (x -np.min(x))/np.mean(x)

def extract_feature(df, train):
    t = group_feature(df, 'ship','x',['max','min','mean','std','skew','sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','x',['count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','y',['max','min','mean','std','skew','sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','v',['max','min','mean','std','skew','sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','d',['max','min','mean','std','skew','sum'])
    train = pd.merge(train, t, on='ship', how='left')
    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['x_max_y_min'] = train['x_max'] - train['y_min']
    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min']==0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']
    
    mode_hour = df.groupby('ship')['hour'].agg(lambda x:x.value_counts().index[0]).to_dict()
    train['mode_hour'] = train['ship'].map(mode_hour)
    
    t = group_feature(df, 'ship','hour',['max','min'])
    train = pd.merge(train, t, on='ship', how='left')
    
    hour_nunique = df.groupby('ship')['hour'].nunique().to_dict()
    date_nunique = df.groupby('ship')['date'].nunique().to_dict()
    train['hour_nunique'] = train['ship'].map(hour_nunique)
    train['date_nunique'] = train['ship'].map(date_nunique)

    t = df.groupby('ship')['time'].agg({'diff_time':lambda x:np.max(x)-np.min(x)}).reset_index()
    t['diff_day'] = t['diff_time'].dt.days
    t['diff_second'] = t['diff_time'].dt.seconds
    train = pd.merge(train, t, on='ship', how='left')
    return train

def extract_dt(df):
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    # df['month'] = df['time'].dt.month
    # df['day'] = df['time'].dt.day
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    # df = df.drop_duplicates(['ship','month'])
    df['weekday'] = df['time'].dt.weekday
    return df

def main_base():
    if config.use_only_test:
        test = pd.read_hdf(feature_path+'test.h5')
        test = extract_dt(test)
        test_label = test.drop_duplicates('ship')
        test_label = extract_feature(test, test_label)
        test_label.to_csv(basefeature_save_path+"test_label.csv",index=None)
    else:
        train = pd.read_hdf(feature_path+'train.h5')
        print(train.shape)
        print(feature_path,train.iloc[5])
        train = extract_dt(train)
        train_label = train.drop_duplicates('ship')
        train_label['type'].value_counts(1)
        type_map = dict(zip(train_label['type'].unique(), np.arange(3)))
        type_map_rev = {v:k for k,v in type_map.items()}
        train_label['type'] = train_label['type'].map(type_map)
        train_label = extract_feature(train, train_label)
        train_label.to_csv(basefeature_save_path+"train_label.csv",index=None)

        features = [x for x in train_label.columns if x not in ['ship','type','time','diff_time','date']]
        target = 'type'

        test = pd.read_hdf(feature_path+'test.h5')
        test = extract_dt(test)
        test_label = test.drop_duplicates('ship')
        test_label = extract_feature(test, test_label)
        test_label.to_csv(basefeature_save_path+"test_label.csv",index=None)

        print(len(features), ','.join(features))

if __name__ == "__main__":
    main_base()
    pass