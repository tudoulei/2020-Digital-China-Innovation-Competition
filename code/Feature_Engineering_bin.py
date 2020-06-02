import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
from scipy.misc import derivative
alpha, gamma = (0.25,1)
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')
from config import config
config = config()
type_map_rev = {0: '拖网', 1: '围网', 2: '刺网'}

model_result = {}
feature_path = "./temp/input_origin/"
bin_feature_save_path = "./temp/binfea/"
use_test = "testB"

os.makedirs(bin_feature_save_path,exist_ok=True)

def get_bin(all_data):
     
    def cut(t):
        c =pd.cut(list(t.v),np.arange(-1,15))

        return (c.value_counts(1)/t[t.v>=0].shape[0]).tolist()

    df = all_data.groupby("ship").apply(cut)

    columns=["cut_"+str(i) for i in range(15)]

    df1 = pd.DataFrame(np.array(df.tolist()),columns=columns)
    
    def cut(t):
        c =pd.cut(list(t.v),np.arange(-1,15))

        return (c.value_counts(1)).tolist()

    df = all_data.groupby("ship").apply(cut)

    columns=["cut_v_"+str(i) for i in range(15)]

    df2 = pd.DataFrame(np.array(df.tolist()),columns=columns)

    def cut(t):
        c =pd.cut(list(t.d),np.arange(-1,360,30))

        return (c.value_counts(1)/t.shape[0]).tolist()

    df = all_data.groupby("ship").apply(cut)

    columns=["cut_d_"+str(i) for i in range(len(np.arange(-1,360,30))-1)]

    df3 = pd.DataFrame(np.array(df.tolist()),columns=columns)
    
    data = pd.concat([df1,df2,df3],axis=1)
    data["ship"] = df.index.tolist()
    return data


def main_bin():
    if config.use_only_test:
        test = pd.read_hdf(feature_path+'test.h5')
        all_data = test
    else:
        train = pd.read_hdf(feature_path+'train.h5')
        get_bin(train).to_csv(bin_feature_save_path+"bin_feature_train.csv",index=None)
        test = pd.read_hdf(feature_path+'test.h5')
        get_bin(test).to_csv(bin_feature_save_path+"bin_feature_test.csv",index=None)
   

if __name__ == "__main__":
    main_bin()
    pass