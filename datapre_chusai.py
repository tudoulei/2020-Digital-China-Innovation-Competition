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

feature_path = config.root_path+"/input/"


train_path = '/tcdata/hy_round1_train_20200102'
train_files = os.listdir(train_path)


def fill(t,mask):
    t.loc[mask,"lat"] =np.nan
    t.loc[mask,"lon"] =np.nan
    t["lat"] = t["lat"].fillna(method="bfill")
    t["lon"] = t["lon"].fillna(method="bfill")
    return t

def time_process(df):
    df['date'] = df['time'].apply(lambda x: x.split()[0])
    df['date'] = df['date'].apply(lambda x: '2019-' + str(x[0:2]) + '-' + str(x[2:4]))
    df['hour'] = df['time'].apply(lambda x: x.split()[1])
    df['time'] = df['date'] + ' ' + df['hour']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')

    return df

def do_something(t):
    t = time_process(t)
    x = t["x"]
    y = t["y"]
    x1 = x/100000+56
    y1 = y/100000-28
    t["x"] = y1.round(3)
    t["y"] = x1.round(3)

    mask =  (t['速度'] >15)

    t.loc[mask,"速度"] =np.nan
    t.loc[mask,"方向"] =np.nan
    t = t.fillna(method="ffill")


    
    t["v_diff"]=t["速度"].diff(1)
    t["d_diff"]=t["方向"].diff(1)
    t["dis_diff"]=np.sqrt(t["x"]**2+t["y"]**2).diff(1)
    t["dis_t_diff"]=np.sqrt(t["x"]**2+t["y"]**2).diff(1)/t["time"].diff().dt.seconds
    t[["v_diff","d_diff"]] = t[["v_diff","d_diff"]].ffill().bfill()
    
    t.drop("hour",axis=1,inplace=True)
    t["d_cos"] = t["方向"].apply(lambda x:np.cos(x/180*np.pi))
    return t


def output(mode="train"):
    if mode == "train":
        filelist = train_files
        filepath = train_path
        # col = ['ship','x','y','v','d','time','type']
        col = ['ship','x','y','v','d','time','type',"date",'v_diff', 'd_diff', 'dis_diff', 'dis_t_diff','d_cos']
        outputfilename = 'train_chusai.h5'


    # main func
    ret = []
    for file in tqdm(filelist):
        df = pd.read_csv(f'{filepath}/{file}')
        # 针对初赛数据的处理
        df = do_something(df)
        df.columns = col
        ret.append(df)

    df = pd.concat(ret)

    os.makedirs(feature_path,exist_ok=True)
    df.to_hdf("./"+outputfilename, 'df', mode='w')

def main_datapre_chusai():
    

    output(mode="train")




if __name__ == "__main__":
    main_datapre_chusai()