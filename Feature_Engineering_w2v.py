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

feature_path = config.root_path+"/input/"

model_save_path = config.root_path+"/model_w2v/"
os.makedirs(model_save_path,exist_ok=1)






def w2c_feature(train,w2c_col,vec_len=100,feature_path="./",sg=0,mode="mean",use_model=False,sample=0,output_feature=0,window=5):

    train[w2c_col+"_gai"] = train[w2c_col].astype("str")

    if use_model:
        print("----加载模型-%s----"%w2c_col)
        try:
            model = Word2Vec.load(model_save_path+'word2vec_'+w2c_col +"_"+str(vec_len)+'.model')
        except:
            print("第二种读取方式")
            model = Word2Vec.load(model_save_path+'word2vec_'+w2c_col+ "_"+mode+"_"+str(vec_len)+'.model')
    else:
  
        print("----建立词向量-%s----"%w2c_col)
        sentences = []
        from tqdm import tqdm
        
   
        sentences = train.groupby("ship")[w2c_col+"_gai"].apply(lambda x: x.tolist()).tolist()


        model = Word2Vec(sentences, size =vec_len,workers=1,seed=1,sg=sg,window=window)
        print("----建立w2c特征-%s----"%w2c_col)
        if mode =="mean":
            model.save(model_save_path+'word2vec_'+w2c_col +"_"+str(vec_len)+'.model')
        else:
            model.save(model_save_path+'word2vec_'+w2c_col+ "_"+mode+"_"+str(vec_len)+'.model')

    if output_feature:
        res = []
        ship_name = []
        from tqdm import tqdm
        print("----输出特征-%s----"%w2c_col)
        for name in tqdm(np.unique(train.ship)):
            df = train[train.ship == name]
            if sample:
                df = df.sample(frac=0.6,random_state=42)
                df = df.sort_values(by='date')
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
        os.makedirs(config.w2v_save_path,exist_ok=1)

        if mode =="mean":
            w2c.to_csv(config.w2v_save_path+"w2c_%s.csv"%w2c_col,index=None)
        else:
            w2c.to_csv(config.w2v_save_path+"w2c_%s_%s.csv"%(mode,w2c_col),index=None)

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

def get_label(all_data,grid_length =100):
    # LON1 = np.min(all_data.x)-0.1
    # LON2 = np.max(all_data.x)+0.1
    # LAT1 = np.min(all_data.y)-0.1
    # LAT2 = np.max(all_data.y)+0.1
    # 尽量用手动值，防止测试集出现错误值
    LON1,LON2,LAT1,LAT2 = (16.435999999999996, 43.683, 105.309, 127.90299999999999)
    label = []
    for x,y in zip(all_data['x'], all_data['y']):
        label.append(generalID(x,y,50,50,LON1,LON2,LAT1,LAT2))
    all_data['label'] = label
    return all_data

def do_something(all_data):
    all_data = all_data.fillna(0)
    # all_data = time_process(all_data)
    all_data = get_label(all_data,grid_length =100)
    # from dis_near import func2
    # dis = func2(all_data)
    # all_data["dis"] = dis
    print("all_data success")
    return all_data



def main_w2v(all_data,output_feature=0,use_model=0):
    
    

    




    all_data = do_something(all_data)


    if 1:
        print("用初赛的model输出词向量")
        # ############################ 初赛的model
        w2c_feature(all_data,w2c_col="v",vec_len=32,feature_path=feature_path,mode="std",use_model=use_model,output_feature=output_feature)
        all_data["d_diff"] = np.rint(all_data["d_diff"]).astype("str")
        w2c_feature(all_data,w2c_col="d_diff",vec_len=32,feature_path=feature_path,use_model=use_model,output_feature=output_feature)
        w2c_feature(all_data,w2c_col="d",vec_len=100,feature_path=feature_path,use_model=use_model,output_feature=output_feature,window=7)
        all_data["v-sg"] = all_data["v"].astype("str")
        w2c_feature(all_data,w2c_col="v-sg",vec_len=100,feature_path=feature_path,sg=1,use_model=use_model,output_feature=output_feature)
        all_data["v-d"] = np.rint(all_data["v"]).astype("str")+"-"+np.rint(all_data["d"]).astype("str")
        w2c_feature(all_data,w2c_col="v-d",vec_len=100,feature_path=feature_path,use_model=use_model,output_feature=output_feature,window=7)

        ############################ 初赛的model
        # d ##############################绝对没问题
        w2c_feature(all_data,w2c_col="v_diff",vec_len=32,feature_path=feature_path,use_model=use_model,output_feature=output_feature)

        # v ############################## 绝对没问题
        w2c_feature(all_data,w2c_col="v",vec_len=32,feature_path=feature_path,mode="min",use_model=use_model,output_feature=output_feature,window=7)
        w2c_feature(all_data,w2c_col="v",vec_len=32,feature_path=feature_path,mode="max",use_model=use_model,output_feature=output_feature)
        # date ##############################没问题
        w2c_feature(all_data,w2c_col="date",vec_len=100,feature_path=feature_path,use_model=use_model,output_feature=output_feature)
        ############################################################
        w2c_feature(all_data,w2c_col="label",vec_len=32,feature_path=feature_path,use_model=use_model,output_feature=output_feature)
        all_data['label-date'] = all_data['label'].astype(str)+"-"+all_data['date'].astype(str)
        w2c_feature(all_data,w2c_col="label-date",vec_len=32,feature_path=feature_path,use_model=use_model,output_feature=output_feature)
        all_data['label-v'] = all_data['label'].astype(str)+"-"+np.rint(all_data['v']).astype(str)
        w2c_feature(all_data,w2c_col="label-v",vec_len=32,feature_path=feature_path,use_model=use_model,output_feature=output_feature)
        all_data['label-d'] = all_data['label'].astype(str)+"-"+ np.rint(all_data['d'] / 30).astype(str)
        w2c_feature(all_data,w2c_col="label-d",vec_len=32,feature_path=feature_path,use_model=use_model,output_feature=output_feature)
        # x-y-v ##############################没问题
        percent = 1000
        w2c_col ="x"
        all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
        w2c_col="y"
        all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
        all_data['x-y-v'] =all_data["x_gai"]+"-"+all_data["y_gai"]+"-"+all_data['v'].astype(str)
        w2c_feature(all_data,w2c_col="x-y-v",vec_len=100,feature_path=feature_path,use_model=use_model,output_feature=output_feature)
        # x-y ##############################百分之百确定100，有两次
        percent = 1000
        w2c_col ="x"
        all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
        w2c_col="y"
        all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
        all_data["x-y-1000"] = all_data["x_gai"]+"-"+all_data["y_gai"]
        w2c_feature(all_data,w2c_col="x-y-1000",vec_len=100,feature_path=feature_path,use_model=use_model,output_feature=output_feature)

        percent = 10
        w2c_col ="x"
        all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
        w2c_col="y"
        all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
        all_data["x-y-10"] = all_data["x_gai"]+"-"+all_data["y_gai"]
        w2c_feature(all_data,w2c_col="x-y-10",vec_len=100,feature_path=feature_path,use_model=use_model,output_feature=output_feature)


    # all_data['label-v'] = all_data['label'].astype(str)+"-"+np.rint(all_data['v']).astype(str)
    # w2c_feature(all_data,w2c_col="label-v",vec_len=32,feature_path=feature_path,use_model=use_model,output_feature=output_feature)

    # all_data["v-sg"] = all_data["v"].astype("str")
    # w2c_feature(all_data,w2c_col="v-sg",vec_len=32,feature_path=feature_path,sg=1,use_model=use_model,output_feature=output_feature)

    # percent = 1000
    # w2c_col ="x"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
    # w2c_col="y"
    # all_data[w2c_col+"_gai"] = np.rint(all_data[w2c_col]*percent).astype("str")
    # all_data["x-y-1000"] = all_data["x_gai"]+"-"+all_data["y_gai"]
    # w2c_feature(all_data,w2c_col="x-y-1000",vec_len=32,feature_path=feature_path,use_model=use_model,output_feature=output_feature)

    # all_data["v-d"] = np.rint(all_data["v"]).astype("str")+"-"+np.rint(all_data["d"]).astype("str")
    # w2c_feature(all_data,w2c_col="v-d",vec_len=32,feature_path=feature_path,use_model=use_model,output_feature=output_feature)


def main_w2v_11():
    # 一起建模
    print("一起建模")
    train = pd.read_hdf(feature_path+'train.h5')
    train_chusai = pd.read_hdf('./train_chusai.h5')
    test = pd.read_hdf(feature_path+'test.h5')
    all_data = pd.concat([train,train_chusai,test],axis=0)
    main_w2v(all_data,output_feature=0,use_model=0)

    # 分别输出-初赛
    print("分别输出-初赛")
    all_data = pd.read_hdf('./train_chusai.h5')
    config.w2v_save_path = config.root_path+"/w2v_chusai/"
    
    main_w2v(all_data,output_feature=1,use_model=1)
    
    # 分别输出-复赛
    print("分别输出-复赛")
    train = pd.read_hdf(feature_path+'train.h5')
    test = pd.read_hdf(feature_path+'test.h5')
    all_data = pd.concat([train,test],axis=0)
    config.w2v_save_path = config.root_path+"/w2v/"
    main_w2v(all_data,output_feature=1,use_model=1)

if __name__ == "__main__":
    main_w2v_11()
    

