import pandas as pd
train_label = pd.read_csv(r"E:\jupyter\智慧海洋\tianchi_ship_2019-master\working\上交\temp\finally_feature/train_label_88.csv")
test_label = pd.read_csv(r"E:\jupyter\智慧海洋\tianchi_ship_2019-master\working\上交\temp\finally_feature/test_label_88.csv")

model_result = {}

base_feature = ['area','diff_second',
 'slope',"c","x","y",'x_max', 'x_min', 'x_mean', 'x_std', 'x_skew', 'x_sum', 'x_count',
       'y_max', 'y_min', 'y_mean', 'y_std', 'y_skew', 'y_sum','x_max_x_min', 'y_max_y_min',
       'y_max_x_min', 'x_max_y_min']

base_feature = ['area','diff_second',
 'slope',"c",'x_max', 'x_min', 'x_mean', 'x_std', 'x_skew',
       'y_max', 'y_min', 'y_mean', 'y_std', 'y_skew', 'y_sum','x_max_x_min', 'y_max_y_min',
       'y_max_x_min', 'x_max_y_min']

model_result["是否添加初始xy特征"] = 3

laji_feature = [
#  'w2c_v_21',
#  'w2c_date_90',
#  'hour_max',
#  'w2c_v_41',
#  'hour_nunique',
#  'weekday',
#  'w2c_d_34',
#  'w2c_date-v_84',
#     "cut_v_15",
#     "cut_v_14"
#  'hour_min'
]
if model_result["是否添加初始xy特征"]==1:
    features = [x for x in train_label.columns if x not in ['ship',"v_min","d_min","d_max",'type','time','diff_time','date','name']]
elif model_result["是否添加初始xy特征"]==2:
    features = [x for x in train_label.columns if x not in laji_feature+['ship',"v_min","d_min","d_max",'type','time','diff_time','date','name']]

else:
    features = [x for x in train_label.columns if x not in base_feature+['ship',"v_min","d_min","d_max",'type','time','diff_time','date','name']]

    
target = 'type'
len(features)
# 1465
train_label[features].head(5).to_csv('4.csv')