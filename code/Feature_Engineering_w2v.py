import os
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from config import config
config = config()

warnings.filterwarnings('ignore')

feature_path = "./temp/input_origin/"
w2v_save_path = "./temp/w2v/"
model_save_path = "temp/model/"
os.makedirs(w2v_save_path,exist_ok=1)
os.makedirs(model_save_path,exist_ok=1)

use_test ="no"

if config.use_only_test:
    w2v_save_path = "./temp/w2v_onlytest/"
    os.makedirs(w2v_save_path,exist_ok=1)

def w2c_feature(train,w2c_col,vec_len=100,feature_path="./",sg=0,mode="mean"):

    train[w2c_col+"_gai"] = train[w2c_col].astype("str")

    if config.use_only_test:
        print("----加载模型-%s----"%w2c_col)
        model = Word2Vec.load(model_save_path+'word2vec_'+w2c_col +"_"+str(vec_len)+'.model')
    
    else:
        print("----建立词向量-%s----"%w2c_col)
        sentences = []
        from tqdm import tqdm
        
        for i in tqdm(np.unique(train.ship)):
            df = train[train.ship == i]
            sentences.append(list(df[w2c_col+"_gai"]))


        model = Word2Vec(sentences, size =vec_len,workers=1,seed=1,sg=sg)
        print("----建立w2c特征-%s----"%w2c_col)
        model.save(model_save_path+'word2vec_'+w2c_col +"_"+str(vec_len)+'.model')


    res = []
    ship_name = []
    from tqdm import tqdm
    print("----输出特征-%s----"%w2c_col)
    for name in tqdm(np.unique(train.ship)):
        df = train[train.ship == name]
        # vec_sum=np.zeros(vec_len)
        vec_sum=[]
        for i in list(df[w2c_col+"_gai"]):
            try:
                vec_sum.append(model[str(i)])
            except:
                pass
        if len(vec_sum) == 0:
            vec_sum = np.zeros((1,vec_len))
        else:
            vec_sum =np.array(vec_sum)
        if mode=="mean":
            res2= np.mean(vec_sum,axis=0)
        elif mode =="std":
            res2= np.std(vec_sum,axis=0)
        elif mode =="min":
            res2= np.min(vec_sum,axis=0)
        elif mode =="max":
            res2= np.max(vec_sum,axis=0)

        ship_name.append(name)
        res.append(res2.tolist())
    res = np.array(res)
        

    if mode =="mean":
        col  =["w2c_"+w2c_col +"_"+str(i) for i in range(vec_len)]
    else:
        col  =["w2c_"+w2c_col +"_"+mode+"_"+str(i) for i in range(vec_len)]
    w2c = pd.DataFrame(columns=col)
    w2c["ship"] = ship_name
    print(res.shape)
    w2c[col]  =res
    if mode =="mean":
        w2c.to_csv(w2v_save_path+"w2c_%s.csv"%w2c_col,index=None)
    else:
        w2c.to_csv(w2v_save_path+"w2c_%s_%s.csv"%(mode,w2c_col),index=None)


def time_process(df):
    df['date'] = df['time'].apply(lambda x: x.split()[0])
    df['date'] = df['date'].apply(lambda x: '2019-' + str(x[0:2]) + '-' + str(x[2:4]))
    df['hour'] = df['time'].apply(lambda x: x.split()[1])
    df['time'] = df['date'] + ' ' + df['hour']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by=['ship', 'time'])
    return df

def generalID(lon,lat,column_num,row_num,LON1,LON2,LAT1,LAT2):
    # 若在范围外的点，返回-1
    #     print(lon,lat)
    if (lon <= LON1) or (lon >= LON2) or (lat <= LAT1) or (lat >= LAT2):
        return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1)/column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1)/row_num
    # 得到二维矩阵坐标索引，并转换为一维ID，即： 列坐标区域（向下取整）+ 1 + 行坐标区域 * 列数
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num


def do_something(all_data):
    all_data = all_data.fillna(0)
    all_data = time_process(all_data)

    LON1 = np.min(all_data.x)*1000-1
    LON2 = np.max(all_data.x)*1000+1
    LAT1 = np.min(all_data.y)*1000-1
    LAT2 = np.max(all_data.y)*1000+1
    # LON1,LON2,LAT1,LAT2 = 5000248.625693836, 7133786.482740336, 3345432.07253926, 7667581.57052392
    label = []
    for x,y in zip(all_data['x'], all_data['y']):
        x=x*1000
        y=y*1000
        label.append(generalID(x,y,50,50,LON1,LON2,LAT1,LAT2))
    all_data['label'] = label
    print("all_data success")
    return all_data



def main_w2v():

    feature_path = "./temp/input_origin/"
    
    if config.use_only_test:
        all_data = pd.read_hdf(feature_path+'test.h5')
    else:
        train = pd.read_hdf(feature_path+'train.h5')
        test = pd.read_hdf(feature_path+'test.h5')
        all_data = pd.concat([train,test],axis=0)

    
    all_data = do_something(all_data)
    
    # # d ##############################绝对没问题
    w2c_feature(all_data,w2c_col="d",vec_len=100,feature_path=feature_path)
    # # # date ##############################没问题
    # w2c_feature(all_data,w2c_col="date",vec_len=100,feature_path=feature_path)
    # # # date-v ##############################没问题 
    # all_data['date-v'] = all_data['date'].astype(str)+"-"+all_data['v'].astype(str)
    # w2c_feature(all_data,w2c_col="date-v",vec_len=100,feature_path=feature_path)
    # ###############################################################
    # all_data['label-date'] = all_data['label'].astype(str)+"-"+all_data['date'].astype(str)
    # w2c_feature(all_data,w2c_col="label-date",vec_len=32,feature_path=feature_path)
    # all_data['label-v'] = all_data['label'].astype(str)+"-"+all_data['v'].astype(str)
    # w2c_feature(all_data,w2c_col="label-v",vec_len=32,feature_path=feature_path)
    # w2c_feature(all_data,w2c_col="label-v",vec_len=32,feature_path=feature_path,mode="std")
    # all_data['label-d'] = all_data['label'].astype(str)+"-"+ np.rint(all_data['d'] / 30).astype(str)
    # w2c_feature(all_data,w2c_col="label-d",vec_len=32,feature_path=feature_path)
    # # # # v ############################## 绝对没问题
    # w2c_feature(all_data,w2c_col="v",vec_len=100,feature_path=feature_path,sg=1)
    # w2c_feature(all_data,w2c_col="v",vec_len=32,feature_path=feature_path,mode="std")
    # w2c_feature(all_data,w2c_col="v",vec_len=32,feature_path=feature_path,mode="min")
    # w2c_feature(all_data,w2c_col="v",vec_len=32,feature_path=feature_path,mode="max")

    # # x-y-v ##############################没问题
    # percent = 1000
    # w2c_col ="x"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
    # w2c_col="y"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
    # all_data['x-y-v'] =all_data["y_gai"]+"-"+all_data["y_gai"]+"-"+all_data['v'].astype(str)
    # w2c_feature(all_data,w2c_col="x-y-v",vec_len=100,feature_path=feature_path)
    # # # x-y ##############################百分之百确定100，有两次
    # percent = 1000
    # w2c_col ="x"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
    # w2c_col="y"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
    # all_data["x-y"] = all_data["x_gai"]+"-"+all_data["y_gai"]
    # w2c_feature(all_data,w2c_col="x-y",vec_len=100,feature_path=feature_path)
    # w2c_feature(all_data,w2c_col="x-y",vec_len=32,feature_path=feature_path,mode="std")
    
    
    


if __name__ == "__main__":
    main_w2v()
    pass
