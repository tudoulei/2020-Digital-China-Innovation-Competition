
import os
import warnings


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import derivative
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from config import config
config = config()


pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')
use_only_test = True
type_map_rev = {0: '拖网', 1: '围网', 2: '刺网'}

finally_feature_path = config.root_path+"/finally_feature/"
os.makedirs(finally_feature_path,exist_ok=1)
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



def get_feature(train_label,test_label):
    base_feature = ['area','diff_second',
    'slope',"c","x","y",'x_max', 'x_min', 'x_mean', 'x_std', 'x_skew', 'x_sum', 'x_count',
        'y_max', 'y_min', 'y_mean', 'y_std', 'y_skew', 'y_sum','x_max_x_min', 'y_max_y_min',
        'y_max_x_min', 'x_max_y_min']

    base_feature = ['area','diff_second',
    'slope',"c",'x_max', 'x_min', 'x_mean', 'x_std', 'x_skew',
        'y_max', 'y_min', 'y_mean', 'y_std', 'y_skew', 'y_sum','x_max_x_min', 'y_max_y_min',
        'y_max_x_min', 'x_max_y_min']

    model_result["是否添加初始xy特征"] = 1

    laji_feature = ["y_max_x_min"]
    # "d_cos","geo_id",

    if model_result["是否添加初始xy特征"]==1:
        features = [x for x in train_label.columns if x not in ['ship',"v_min","d_min","d_max",'type','time','diff_time','date','name']]
    elif model_result["是否添加初始xy特征"]==2:
        features = [x for x in train_label.columns if x not in laji_feature+['ship',"v_min","d_min","d_max",'type','time','diff_time','date','name']]

    else:
        features = [x for x in train_label.columns if x not in base_feature+['ship',"v_min","d_min","d_max",'type','time','diff_time','date','name']]
    target = "type"
    print("特征数量",len(features))
    # print(features)
    return features,target



def get_fold_index(train_label,test_label,seed=42):
    train_label_len = train_label.shape[0]
    data =pd.concat([train_label,test_label])
  
    # LON1 = np.min(data.x_mean)-1
    # LON2 = np.max(data.x_mean)+1
    # LAT1 = np.min(data.y_mean)-1
    # LAT2 = np.max(data.y_mean)+1
    LON1,LON2,LAT1,LAT2 = (16.435999999999996, 43.683, 105.309, 127.90299999999999)
    data['label'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    
    len(np.unique(data['label']))

    train_label["label"] = data['label'][:train_label_len]

    len(np.unique(train_label["label"]))
    
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
    for i in np.unique(train_label['label']): 
        x=zong_index[train_label["label"][mask]==i]
        y=train_label[mask][train_label["label"][mask]==i]["type"]
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


# 初赛预测模型
def predict_chusai_model(train_label,test_label):




    lightgbm_model_path = "./model_lgb_chusai/"
    # lightgbm_model_path = "../../tianchi_ship_2019-master/working/fuxian/temp/model_lg/"


    data =pd.concat([train_label,test_label])


    LAT1= np.min(data.x)-0.1
    LAT2 = np.max(data.x)+0.1
    LON1 = np.min(data.y)-0.1
    LON2= np.max(data.y)+0.1
    data["geo_id"] = data.apply(lambda x: generalID(x['y_mean'], x['x_mean'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)

    import lightgbm as lgb
    import time
    pred = np.zeros((len(data),3))
    
    print("------------------初赛模型预测--------------------")
    for i in os.listdir(lightgbm_model_path):
        model = lgb.Booster(model_file=lightgbm_model_path+i)
        # data[model.feature_name()].to_csv("1.csv")
        
        test_pred = model.predict(data[model.feature_name()])
        pred += test_pred/5
    pred_ = np.argmax(pred, axis=1)
    print(pd.DataFrame(pred_)[0].value_counts(1))
    # all_val_f1 = metrics.f1_score(pred_[:-2], data["type"][:-2], average='macro')
    # print('oof f1', all_val_f1)

    type_map_rev = {0: '拖网', 1: '围网', 2: '刺网'}

    sub = pd.DataFrame({
        "ship":data["ship"],
        "pred1":pred[:,0],
        "pred2":pred[:,1],
        "pred3":pred[:,2]
    })
    sub_feature_path = "./temp/sub_pred/"
    os.makedirs(sub_feature_path,exist_ok=1)
    sub.to_csv(sub_feature_path+"sub.csv")
    train_label = pd.merge(train_label,sub,how="left",on="ship")
    test_label = pd.merge(test_label,sub,how="left",on="ship")


    return train_label,test_label


def output_finally_feature(name=None):

    ##########################  基础特征   ########################################
    feature_path = config.root_path+"/basefea/"
    print(feature_path)
    train_label=pd.read_csv( feature_path+"train_label.csv")
    test_label=pd.read_csv( feature_path+"test_label.csv")
    print(train_label.columns)

    ##########################  w2v特征   ########################################
    feature_path = config.root_path+"/w2v/"
    if name is not None:
        path = name
    else:
        path = os.listdir(feature_path)
    for i in path:
        print(i)
        feature = pd.read_csv(feature_path +i)
        train_label = pd.merge(train_label,feature,how="left",on="ship")
        test_label = pd.merge(test_label,feature,how="left",on="ship")

    

    ##########################  分箱特征   ########################################
    bin_feature_save_path = config.root_path+"/binfea/"
    
    train_feature = pd.read_csv(bin_feature_save_path+"bin_feature_train.csv")
    test_feature= pd.read_csv(bin_feature_save_path+"bin_feature_test.csv")
    train_label = pd.merge(train_label,train_feature,how="left",on="ship")
    test_label = pd.merge(test_label,test_feature,how="left",on="ship")
    ##########################  输出   ########################################



    data =pd.concat([train_label,test_label])
    LON1,LON2,LAT1,LAT2 = (16.435999999999996, 43.683, 105.309, 127.90299999999999)
    data['geoid_4'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],4,4,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_6'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],6,6,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_10'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_50'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],50,50,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_100'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],100,100,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_100'] = data.apply(lambda x: generalID(x['x_max'], x['y_max'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_10_max'] = data.apply(lambda x: generalID(x['x_max'], x['y_max'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_10_min'] = data.apply(lambda x: generalID(x['x_min'], x['y_min'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_x_mean_v_mean'] = data.\
    apply(lambda x: generalID(x['x_mean'], x['v_mean'],20,10,LON1,LON2,LAT1,LAT2), axis = 1)

    
    
    col = ["ship","geoid_10","geoid_50","geoid_100","geoid_10_max","geoid_10_min",
    "geoid_x_mean_v_mean"
    
    ]

    train_label = pd.merge(train_label,data[col],how="left",on="ship")
    test_label = pd.merge(test_label,data[col],how="left",on="ship")



    # 如果用初赛模型预测新的三组概率
    if config.use_chusai_model:
        train_label,test_label = predict_chusai_model(train_label,test_label)



    if config.use_disnear:
        from dis_near import func1
        data =pd.concat([train_label,test_label])
        dis = func1(data)
        data["dis"] = dis
        train_label = pd.merge(train_label,data[["ship","dis"]],how="left",on="ship")
        test_label = pd.merge(test_label,data[["ship","dis"]],how="left",on="ship")



    train_label.to_csv(finally_feature_path+"train_label_88.csv",index=None)
    test_label.to_csv(finally_feature_path+"test_label_88.csv",index=None)

def output_finally_feature_chusai(name=None):

    ##########################  基础特征   ########################################
    feature_path = config.root_path+"/basefea/"
    print(feature_path)
    train_label=pd.read_csv( feature_path+"train_label_chusai.csv")
    
    print(train_label.columns)

    ##########################  w2v特征   ########################################
    feature_path = config.root_path+"/w2v_chusai/"
    if name is not None:
        path = name
    else:
        path = os.listdir(feature_path)
    for i in path:
        print(i)
        feature = pd.read_csv(feature_path +i)
        train_label = pd.merge(train_label,feature,how="left",on="ship")
        

    

    ##########################  分箱特征   ########################################
    bin_feature_save_path = config.root_path+"/binfea/"
    
    train_feature = pd.read_csv(bin_feature_save_path+"bin_feature_train_chusai.csv")
    train_label = pd.merge(train_label,train_feature,how="left",on="ship")

    ##########################  输出   ########################################



    data =train_label.copy()
    LON1,LON2,LAT1,LAT2 = (16.435999999999996, 43.683, 105.309, 127.90299999999999)
    data['geoid_4'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],4,4,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_6'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],6,6,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_10'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_50'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],50,50,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_100'] = data.apply(lambda x: generalID(x['x_mean'], x['y_mean'],100,100,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_100'] = data.apply(lambda x: generalID(x['x_max'], x['y_max'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_10_max'] = data.apply(lambda x: generalID(x['x_max'], x['y_max'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_10_min'] = data.apply(lambda x: generalID(x['x_min'], x['y_min'],10,10,LON1,LON2,LAT1,LAT2), axis = 1)
    data['geoid_x_mean_v_mean'] = data.\
    apply(lambda x: generalID(x['x_mean'], x['v_mean'],20,10,LON1,LON2,LAT1,LAT2), axis = 1)
    
    col = ["ship","geoid_10","geoid_50","geoid_100","geoid_10_max","geoid_10_min",
    "geoid_x_mean_v_mean"
    
    ]

    train_label = pd.merge(train_label,data[col],how="left",on="ship")
    

    train_label.to_csv(finally_feature_path+"train_label_88_chusai.csv",index=None)



def other(train_label):
    # for i in os.listdir("temp/w2v"):
    for w2c_col in ["v",'d','label-v','x-y-1000']:
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
    # output_finally_feature()
    output_finally_feature_chusai()

    