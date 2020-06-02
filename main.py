import numpy as np
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



def submit():
    # from get_all_feature  import output_finally_test_feature
    from datapre import main_datapre
    from Feature_Engineering_base import main_base
    from Feature_Engineering_w2v import main_w2v,main_w2v_11
    from Feature_Engineering_bin import main_bin
    from get_all_feature import get_feature,get_fold_index,output_finally_feature_chusai
    from model import submit_file,train
    if config.online:
        from datapre_chusai import main_datapre_chusai
        main_datapre_chusai()
        main_bin(mode="chusai")
        main_base(mode="chusai")
        

    if config.online:
        main_datapre()
        main_w2v_11()
        main_base(mode="fusai")
        main_bin(mode="fusai")
        

    output_finally_feature_chusai()
    output_finally_feature()
    feature_path = config.root_path+"/finally_feature/"
    
    if config.mode == "train":
        train_label=pd.read_csv(feature_path+"train_label_88.csv")
        
        test_label=pd.read_csv(feature_path+"test_label_88.csv")
        print(feature_path)
        features,target= get_feature(train_label,test_label)
        index_list,index_list_val= get_fold_index(train_label,test_label)
        num_fu = train_label.shape[0]
        train_label_chusai=pd.read_csv(feature_path+"train_label_88_chusai.csv")
        train_label.head(5).to_csv("1-1.csv")
        train_label_chusai.head(5).to_csv("1-2.csv")
        train_label = pd.concat([train_label,train_label_chusai]).reset_index(drop=True)
        num_chu = train_label_chusai.shape[0]
        
        print(train_label.shape)
        index_list["1"] = np.append(index_list["1"],np.arange(num_fu,num_fu+num_chu))
        index_list["2"] = np.append(index_list["2"],np.arange(num_fu,num_fu+num_chu))
        index_list["3"] = np.append(index_list["3"],np.arange(num_fu,num_fu+num_chu))
        index_list["4"] = np.append(index_list["4"],np.arange(num_fu,num_fu+num_chu))
        index_list["5"] = np.append(index_list["5"],np.arange(num_fu,num_fu+num_chu))

        train(index_list,index_list_val,train_label,test_label,features,target)
    elif config.mode == "submit":
        test_label=pd.read_csv(feature_path+"test_label_88.csv")
        submit_file(test_label,features)

if __name__ == "__main__":

    submit()

    