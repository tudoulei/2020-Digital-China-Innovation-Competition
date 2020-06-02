from model import train
from get_all_feature import output_finally_feature,get_feature,get_fold_index
import pandas as pd
import time
import sys
import os
from config import config
config = config()

os.makedirs("./log",exist_ok=1)
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
 
sys.stdout = Logger("./log/"+time.strftime('%Y-%m-%d-%H-%M-%S')+".txt")

def train_model():
    # 整理特征
    output_finally_feature()
    # 打开数据
    feature_path = "./temp/finally_feature/"
    train_label=pd.read_csv(feature_path+"train_label_88.csv")
    test_label=pd.read_csv(feature_path+"test_label_88.csv")
    # 获取X，y标签
    features,target= get_feature()
    # 按地理编码五折
    index_list,index_list_val = get_fold_index(train_label,test_label)
    # 训练
    train(index_list,index_list_val,train_label,test_label,features,target)
    pass

def submit():
    from get_all_feature  import output_finally_test_feature
    from datapre import main_datapre
    from Feature_Engineering_base import main_base
    from Feature_Engineering_w2v import main_w2v
    from Feature_Engineering_bin import main_bin
    from get_all_feature import get_feature,get_fold_index
    from model import submit_file,train,train2
    main_datapre()
    main_base()
    main_w2v()
    main_bin()
    output_finally_feature()
    feature_path = "./temp/finally_feature/"
    
    if config.mode == "train":
        train_label=pd.read_csv(feature_path+"train_label_88.csv")
        test_label=pd.read_csv(feature_path+"test_label_88.csv")
        features,target= get_feature()
        index_list,index_list_val= get_fold_index(train_label,test_label)
        train(index_list,index_list_val,train_label,test_label,features,target)
    elif config.mode == "submit":
        test_label=pd.read_csv(feature_path+"test_label_88.csv")
        submit_file(test_label,features)

if __name__ == "__main__":
    # train_model()
    submit()

    