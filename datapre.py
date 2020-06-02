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

# from config import config_chusai
# config = config_chusai()
feature_path = config.root_path+"/input/"



train_files = os.listdir(config.train_path)
if config.testA_path is not None:
    testA_files = os.listdir(config.testA_path) 
if config.testB_path is not None:
    testB_files = os.listdir(config.testB_path)
# print(len(train_files), len(testA_files))


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


    


    mask =  (t['速度'] >15)

    t.loc[mask,"速度"] =np.nan
    t.loc[mask,"方向"] =np.nan
    t = t.fillna(method="ffill")

    mask =  (t['lat'] > 16 ) & (t['lon'] >100)
    # t= t[mask]
    t.loc[~mask,"lat"] =np.nan
    t.loc[~mask,"lon"] =np.nan
    t["lat"] = t["lat"].fillna(method="bfill")
    t["lon"] = t["lon"].fillna(method="bfill")
    t["lat"] = t["lat"].ffill().bfill()
    t["lon"] = t["lon"].ffill().bfill()


    def fill(t,mask):
        t.loc[mask,"lat"] =np.nan
        t.loc[mask,"lon"] =np.nan
        t["lat"] = t["lat"].fillna(method="bfill")
        t["lon"] = t["lon"].fillna(method="bfill")
        return t

    # 方案1，这是一个突然变小的错误数
    shift = (t.shift(1)["lat"]-t["lat"]) >4  # 做差后值是否过大
    error= np.mean(t["lat"])+2*np.std(t["lat"])  # 这个数是否超过阈值

    mask = (shift)  & (t["lat"] <error)
    t=fill(t,mask)
    # t=t[~mask]
    # 方案1，这是一个突然增大的错误数
    shift = (t.shift(1)["lat"]-t["lat"]) <-4  # 做差后值是否过大
    error= np.mean(t["lat"])-2*np.std(t["lat"])  # 这个数是否超过阈值

    mask = (shift) & (t["lat"] >error)
    t=fill(t,mask)
    # t=t[~mask]

    
    t["v_diff"]=t["速度"].diff(1)
    t["d_diff"]=t["方向"].diff(1)
    t["dis_diff"]=np.sqrt(t["lon"]**2+t["lat"]**2).diff(1)
    t["dis_t_diff"]=np.sqrt(t["lon"]**2+t["lat"]**2).diff(1)/t["time"].diff().dt.seconds
    t[["v_diff","d_diff"]] = t[["v_diff","d_diff"]].ffill().bfill()
    
    t.drop("hour",axis=1,inplace=True)
    t["d_cos"] = t["方向"].apply(lambda x:np.cos(x/180*np.pi))
    return t


def output(mode="train"):
    if mode == "train":
        filelist = train_files
        filepath = config.train_path
        # col = ['ship','x','y','v','d','time','type']
        col = ['ship','x','y','v','d','time','type',"date",'v_diff', 'd_diff', 'dis_diff', 'dis_t_diff','d_cos']
        outputfilename = 'train.h5'
    elif mode == "testA":
        filelist = testA_files
        filepath = config.testA_path
        # col =['ship','x','y','v','d','time']
        col = ['ship','x','y','v','d','time',"date",'v_diff', 'd_diff', 'dis_diff', 'dis_t_diff','d_cos']
        outputfilename = 'test.h5'
    elif mode == "testB":
        filelist = testB_files
        filepath = config.testB_path
        col = ['ship','x','y','v','d','time',"date",'v_diff', 'd_diff', 'dis_diff', 'dis_t_diff','d_cos']
        outputfilename = 'test.h5'

    
    # main func
    ret = []
    for file in tqdm(filelist):
        df = pd.read_csv(f'{filepath}/{file}')
        # 针对初赛数据的处理
        if config.train_path == '/tcdata/hy_round1_train_20200102':
            df.columns = ['渔船ID', 'lat', 'lon', '速度', '方向', 'time', 'type']
        df = do_something(df)
        df.columns = col
        ret.append(df)


    df = pd.concat(ret)

    os.makedirs(feature_path,exist_ok=True)
    df.to_hdf(feature_path+outputfilename, 'df', mode='w')
    df.to_csv(feature_path+"train.csv")

def main_datapre():
    
    if not config.use_only_test:
        output(mode="train")
    if config.use_test:
        if config.test_mode == "testA":
            output(mode="testA")
        elif config.test_mode == "testB":
            output(mode="testB")
        else:
            print("缺少test数据")

if __name__ == "__main__":
    main_datapre()