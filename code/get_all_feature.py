
import os
import warnings


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import derivative
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')
use_only_test = True
type_map_rev = {0: '拖网', 1: '围网', 2: '刺网'}

model_result = {}


def generalID(lon,lat,column_num,row_num,LON1,LON2,LAT1,LAT2):
    # 若在范围外的点，返回-1
    #     print(lon,lat)
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数等分切割
    column = (LON2 - LON1)/column_num
    # 把纬度范围根据行数数等分切割
    row = (LAT2 - LAT1)/row_num
    # 得到二维矩阵坐标索引，并转换为一维ID，即： 列坐标区域（向下取整）+ 1 + 行坐标区域 * 列数
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num



def get_feature():
    feature_path = "./temp/finally_feature/"
    if not use_only_test:
        train_label = pd.read_csv(feature_path+"train_label_88.csv")
    else:
        train_label = pd.read_csv(feature_path+"test_label_88.csv")

    base_feature = ['area','diff_second',
    'slope',"c","x","y",'x_max', 'x_min', 'x_mean', 'x_std', 'x_skew', 'x_sum', 'x_count',
        'y_max', 'y_min', 'y_mean', 'y_std', 'y_skew', 'y_sum','x_max_x_min', 'y_max_y_min',
        'y_max_x_min', 'x_max_y_min']

    base_feature = ['area','diff_second',
    'slope',"c",'x_max', 'x_min', 'x_mean', 'x_std', 'x_skew',
        'y_max', 'y_min', 'y_mean', 'y_std', 'y_skew', 'y_sum','x_max_x_min', 'y_max_y_min',
        'y_max_x_min', 'x_max_y_min']

    model_result["是否添加初始xy特征"] = 1

    laji_feature = ["x_min","y_min"]
    # "d_cos","geo_id",

    if model_result["是否添加初始xy特征"]==1:
        features = [x for x in train_label.columns if x not in ['ship',"v_min","d_min","d_max",'type','time','diff_time','date','name']]
    elif model_result["是否添加初始xy特征"]==2:
        features = [x for x in train_label.columns if x not in laji_feature+['ship',"v_min","d_min","d_max",'type','time','diff_time','date','name']]

    else:
        features = [x for x in train_label.columns if x not in base_feature+['ship',"v_min","d_min","d_max",'type','time','diff_time','date','name']]
    target = "type"
    print("特征数量",len(features))
    return features,target



def get_fold_index(train_label,test_label,seed=42):
    train_label_len = train_label.shape[0]
    data =pd.concat([train_label,test_label])
    LON1 = np.min(data.x)-1
    LON2 = np.max(data.x)+1
    LAT1 = np.min(data.y)-1
    LAT2 = np.max(data.y)+1
    data['label'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    
    len(np.unique(data['label']))

    train_label["geo_id"] = data['label'][:train_label_len]

    len(np.unique(train_label["geo_id"]))
    
    index_list = dict()
    index_list["1"] = np.array([])
    index_list["2"] = np.array([])
    index_list["3"] = np.array([])
    index_list["4"] = np.array([])
    index_list["5"] = np.array([])

    index_list_val = dict()
    index_list_val["1"] = np.array([])
    index_list_val["2"] = np.array([])
    index_list_val["3"] = np.array([])
    index_list_val["4"] = np.array([])
    index_list_val["5"] = np.array([])
    mask = np.ones(train_label_len,dtype=bool)
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    zong_index= np.arange(train_label_len)[mask]
    for i in np.unique(train_label['geo_id']): 
        x=zong_index[train_label["geo_id"][mask]==i]
        y=train_label[mask][train_label["geo_id"][mask]==i]["type"]
    #     print(len(y))
        if len(y)<10:
            print(":error")
            for index in range(5):
                index_list[str(index+1)] = np.append(index_list[str(index+1)],x)
        else:
            for index, (train_idx, val_idx) in enumerate(fold.split(x, y)):

                index_list[str(index+1)] = np.append(index_list[str(index+1)],x[train_idx])
                index_list_val[str(index+1)] = np.append(index_list_val[str(index+1)],x[val_idx])
    return index_list,index_list_val

def get_fold_index2(train_label,seed=42):

    data =train_label
    LON1 = np.min(data.x)-1
    LON2 = np.max(data.x)+1
    LAT1 = np.min(data.y)-1
    LAT2 = np.max(data.y)+1
    data['label'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    
    len(np.unique(data['label']))

    train_label["geo_id"] = data['label']


    index_list = dict()
    index_list["1"] = np.array([])
    index_list["2"] = np.array([])
    index_list["3"] = np.array([])
    index_list["4"] = np.array([])
    index_list["5"] = np.array([])

    index_list_val = dict()
    index_list_val["1"] = np.array([])
    index_list_val["2"] = np.array([])
    index_list_val["3"] = np.array([])
    index_list_val["4"] = np.array([])
    index_list_val["5"] = np.array([])
    mask = np.ones(train_label.shape[0],dtype=bool)
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    zong_index= np.arange(train_label.shape[0])[mask]
    for i in np.unique(train_label['geo_id']): 
        x=zong_index[train_label["geo_id"][mask]==i]
        y=train_label[mask][train_label["geo_id"][mask]==i]["type"]
    #     print(len(y))
        if len(y)<10:
            print(":error")
            for index in range(5):
                index_list[str(index+1)] = np.append(index_list[str(index+1)],x)
        else:
            for index, (train_idx, val_idx) in enumerate(fold.split(x, y)):

                index_list[str(index+1)] = np.append(index_list[str(index+1)],x[train_idx])
                index_list_val[str(index+1)] = np.append(index_list_val[str(index+1)],x[val_idx])
    return index_list,index_list_val


use_test = "no"
def output_finally_feature():

    ##########################  基础特征   ########################################
    feature_path = "./temp/basefea/"
    model_result["basefeature_path"] =feature_path
    train_label=pd.read_csv( feature_path+"train_label.csv")
    test_label=pd.read_csv( feature_path+"test_label.csv")


    ##########################  w2v特征   ########################################
    root_path = "./temp/w2v/"
    for i in os.listdir(root_path):
        print(i)
        feature = pd.read_csv(root_path +i)
        train_label = pd.merge(train_label,feature,how="left",on="ship")
        test_label = pd.merge(test_label,feature,how="left",on="ship")
    # root_path = "./temp/w2v_onlytest/"
    # for i in os.listdir(root_path):
    #     print(i)
    #     feature = pd.read_csv(root_path +i)
    #     test_label = pd.merge(test_label,feature,how="left",on="ship")
    

    ##########################  分箱特征   ########################################
    bin_feature_save_path = "./temp/binfea/"
    
    train_feature = pd.read_csv(bin_feature_save_path+"bin_feature_train.csv")
    test_feature= pd.read_csv(bin_feature_save_path+"bin_feature_test.csv")
    train_label = pd.merge(train_label,train_feature,how="left",on="ship")
    test_label = pd.merge(test_label,test_feature,how="left",on="ship")
    ##########################  输出   ########################################
    feature_path = "./temp/finally_feature/"
    os.makedirs(feature_path,exist_ok=1)


    # data =pd.concat([train_label,test_label])
    # LON1,LON2,LAT1,LAT2 = 5000248.625693836, 7133786.482740336, 3345432.07253926, 7667581.57052392
    # data['label'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],100,100,LON1,LON2,LAT1,LAT2), axis = 1)
    # train_label["geo_id"] = data['label'][:7000]
    # test_label["geo_id"] = data['label'][7000:]
    # train_label,test_label = add_geoid(train_label,test_label)

    train_label.to_csv(feature_path+"train_label_88.csv",index=None)
    test_label.to_csv(feature_path+"test_label_88.csv",index=None)

def output_finally_test_feature():

    ##########################  基础特征   ########################################
    feature_path = "./temp/basefea/"
    model_result["basefeature_path"] =feature_path
    if use_test == "testB":
        print("---------------使用testB------------------------")
        test_label=pd.read_csv( feature_path+"testB_label.csv")
    elif use_test == "testA":
        print("---------------使用testA------------------------")
        test_label=pd.read_csv( feature_path+"testA_label.csv")
    else:
        test_label=pd.read_csv( feature_path+"train_label.csv")


    ##########################  w2v特征   ########################################
    root_path = "./temp/w2v/"
    for i in os.listdir(root_path):
        print(i)
        feature = pd.read_csv(root_path +i)
        test_label = pd.merge(test_label,feature,how="left",on="ship")
        
    

    ##########################  分箱特征   ########################################
    bin_feature_save_path = "./temp/binfea/"
    bindata=pd.read_csv(bin_feature_save_path+"bin_feature.csv")

    columns  = bindata.columns
    
    test_label[columns] = bindata[columns]
    ##########################  输出   ########################################
    feature_path = "./temp/finally_feature/"
    os.makedirs(feature_path,exist_ok=1)

    LON1 = np.min(test_label.x)-1
    LON2 = np.max(test_label.x)+1
    LAT1 = np.min(test_label.y)-1
    LAT2 = np.max(test_label.y)+1
    test_label['geo_id'] = test_label.apply(lambda x: generalID(x['x_mean'], x['y_mean'],5,5,LON1,LON2,LAT1,LAT2), axis = 1)


    test_label.to_csv(feature_path+"test_label_88.csv",index=None)


def add_geoid(train_label,test_label):
    data =pd.concat([train_label,test_label])
    LON1 = np.min(data.x)-1
    LON2 = np.max(data.x)+1
    LAT1 = np.min(data.y)-1
    LAT2 = np.max(data.y)+1
    data['label'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],5,5,LON1,LON2,LAT1,LAT2), axis = 1)
    train_label["geo_id"] = data['label'][:7000]
    test_label["geo_id"] = data['label'][7000:]
    return train_label,test_label
def other(train_label):
    # for i in os.listdir("temp/w2v"):
    for w2c_col in ["d",'date-v','date','label-v','v-d','v','x-y-v','x-y']:
        print(w2c_col)
        train_label[w2c_col+"_mean"] = np.mean(train_label[["w2c_"+w2c_col +"_"+str(i)  for i in range(100)]],axis=1)
        train_label[w2c_col+"_std"] = np.std(train_label[["w2c_"+w2c_col +"_"+str(i)  for i in range(100)]],axis=1)
        train_label[w2c_col+"_min"] = np.min(train_label[["w2c_"+w2c_col +"_"+str(i)  for i in range(100)]],axis=1)
        train_label[w2c_col+"_max"] = np.max(train_label[["w2c_"+w2c_col +"_"+str(i)  for i in range(100)]],axis=1)
    for w2c_col in ["x-y-v-1000"]:
        train_label[w2c_col+"_mean"] = np.mean(train_label[["w2c_"+w2c_col +"_"+str(i)  for i in range(200)]],axis=1)
        train_label[w2c_col+"_std"] = np.std(train_label[["w2c_"+w2c_col +"_"+str(i)  for i in range(200)]],axis=1)
        train_label[w2c_col+"_min"] = np.min(train_label[["w2c_"+w2c_col +"_"+str(i)  for i in range(200)]],axis=1)
        train_label[w2c_col+"_max"] = np.max(train_label[["w2c_"+w2c_col +"_"+str(i)  for i in range(200)]],axis=1)


    return train_label

if __name__ == "__main__":
    output_finally_feature()
    