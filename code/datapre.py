import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')
from config import config
config = config()

train_files = os.listdir(config.train_path)
if config.testA_path is not None:
    testA_files = os.listdir(config.testA_path) 
if config.testB_path is not None:
    testB_files = os.listdir(config.testB_path)
# print(len(train_files), len(testA_files))



def do_something_way1(t):
    mask =  (t['速度'] >15)

    t.loc[mask,"速度"] =np.nan
    t.loc[mask,"方向"] =np.nan
    t = t.fillna(method="ffill")
    
    return t

def do_something_way2(t):
    mask =  (t['速度'] >15)

    t.loc[mask,"速度"] =np.nan
    t.loc[mask,"方向"] =np.nan
    t = t.fillna(method="ffill")
    
    t["方向"] = t["方向"].apply(lambda x:np.cos(x/180*np.pi))
    
    return t


def output(mode="train",way="way1"):
    if mode == "train":
        filelist = train_files
        filepath = config.train_path
        col = ['ship','x','y','v','d','time','type']
        outputfilename = 'train.h5'
    elif mode == "testA":
        filelist = testA_files
        filepath = config.testA_path
        col =['ship','x','y','v','d','time']
        outputfilename = 'test.h5'
    elif mode == "testB":
        filelist = testB_files
        filepath = config.testB_path
        col =['ship','x','y','v','d','time']
        outputfilename = 'test.h5'
    if way=="way1":
        feature_path = "./temp/input_origin/"
    elif way=="way2":
        feature_path = "./temp/input/"
    # main func
    ret = []
    for file in tqdm(filelist):
        df = pd.read_csv(f'{filepath}/{file}')
        if way=="way1":df = do_something_way1(df)
        elif way=="way2":df = do_something_way2(df)
        ret.append(df)
    df = pd.concat(ret)
    df.columns = col
    os.makedirs(feature_path,exist_ok=True)
    df.to_hdf(feature_path+outputfilename, 'df', mode='w')

def main_datapre():
    if not config.use_only_test:
        output(mode="train",way="way1")
        output(mode="train",way="way2")
        # pass
    if config.use_test:
        if config.test_mode == "testA":
            output(mode="testA",way="way1")
            output(mode="testA",way="way2")
        elif config.test_mode == "testB":
            output(mode="testB",way="way1")
            output(mode="testB",way="way2")
        else:
            print("缺少test数据")

if __name__ == "__main__":
    main_datapre()