from sklearn.metrics import f1_score
import lightgbm as lgb
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import derivative
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time
from config import config
config = config()
alpha, gamma = (0.25,1)
type_map_rev = {0: '拖网', 1: '围网', 2: '刺网'}

lightgbm_model_path = "./temp/model_lg/"
os.makedirs(lightgbm_model_path,exist_ok=1)
os.makedirs("submit/",exist_ok=1)
model_result = {}

def focal_loss_lgb(y_pred, dtrain, alpha=0.25, gamma=2, num_class=3):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    # N observations x num_class arrays
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1,num_class, order='F')
    # alpha and gamma multiplicative factors with BCEWithLogitsLoss
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    # flatten in column-major (Fortran-style) order
    return grad.flatten('F'), hess.flatten('F')


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha=0.25, gamma=2, num_class=3):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class, order='F')
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    # a variant can be np.sum(loss)/num_class
    return 'focal_loss', np.mean(loss), False


def lgb_f1_score2(pred, data_vail):
    labels = data_vail.get_label()
    pred = np.argmax(pred.reshape(3, -1), axis=0)      # lgb的predict输出为各类型概率值
    score_vail = f1_score(y_true=labels, y_pred=pred, average='macro')
    return 'f1', score_vail, True

def lgb_f1_score(pred, data_vail):
    labels = data_vail.get_label()
    pred = np.argmax(pred.reshape(3, -1), axis=0)      # lgb的predict输出为各类型概率值
    labels = (labels == 1).astype(int)
    pred =( pred == 1).astype(int)
    score_vail = f1_score(y_true=labels, y_pred=pred)
    return 'f1', score_vail, True

def lgb_f1_score_at_1(pred, data_vail):
    labels = data_vail.get_label()
    pred = np.argmax(pred.reshape(3, -1), axis=0)      # lgb的predict输出为各类型概率值
    labels = labels.astype(int)
    pred = pred.astype(int)
    # labels = (labels == 1).astype(int)
    # pred =( pred == 1).astype(int)
    score_vail = f1_score(y_true=labels, y_pred=pred,average="macro")
    return 'f1', score_vail, True


def feature_importance(models):
    ret = []
    for index, model in enumerate(models):
        df = pd.DataFrame()
        df['name'] = model.feature_name()
        df['score'] = model.feature_importance()
        df['fold'] = index
        ret.append(df)
        
    df = pd.concat(ret)

    df = df.groupby('name', as_index=False)['score'].mean()
    df = df.sort_values(['score'], ascending=False)

    df['name'][:10]

    df.to_csv('feature.csv')


def train(index_list,index_list_val,train_label,test_label,features,target):
    
     
    train_label[features].head(5).to_csv("kankan.csv")
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_val_f1 = []

    use_focalloss = 1
    focal_loss = lambda x,y: focal_loss_lgb(x, y, 0.25, 2., 3)
    eval_error = lambda x,y: focal_loss_lgb_eval_error(x, y, 0.25, 2., 3)
    

     
    params = {
        'n_estimators': config.epoch,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
    #     "min_data_in_leaf":10,
        'learning_rate': config.learning_rate, 
    #     "bagging_fraction":0.9,
    #     "feature_fraction":0.9,
    #     "feature_freq":10,
    #     "bagging_freq":10,
    #     "lambda_l1":0.8,
    #     "lambda_l2":0.8,
    #     "metric":lgb_f1_score,
    #     "max_depth":6,
    #     'num_leaves': 6, 
    #     "label_weights_":list(train_label['type'].value_counts(1))*10,
    #     "imbalance":True,
        'num_class': 3,
    #     'num_leaves':60,
        'early_stopping_rounds': 50,
    }
    model_result["use mask"] = 1
    if model_result["use mask"]:
        X = train_label[features].copy()
        y = train_label[target]
    else:
        X = train_label[features].copy()
        y = train_label[target]
    models = []
    pred = np.zeros((len(test_label),3))
    oof = np.zeros((len(X), 3))
    evals_result = {}  #记录训练结果所用
    error_val_list =[]
    for index in range(5):
        train_idx, val_idx = index_list[str(index+1)].astype(int),index_list_val[str(index+1)].astype(int)
        train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])
    # feval=lgb_f1_score, 
    # fobj=focal_loss, feval=eval_error,
        if use_focalloss ==1:
            model = lgb.train(params, train_set, fobj=focal_loss, feval=eval_error,
                        valid_sets=[train_set, val_set], verbose_eval=300)
        elif use_focalloss ==2:
            model = lgb.train(params, train_set, fobj=focal_loss, feval=lgb_f1_score,
                        valid_sets=[val_set], verbose_eval=300)
        else:
            model = lgb.train(params, train_set,evals_result=evals_result,feval=lgb_f1_score_at_1, 
                        valid_sets=[ val_set], verbose_eval=300)
        models.append(model)
        val_pred = model.predict(X.iloc[val_idx])
        oof[val_idx] = val_pred
        val_y = y.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        val_f1 = metrics.f1_score(val_y, val_pred, average='macro')
        all_val_f1.append(val_f1)
        print(index, 'val f1', val_f1)
        
        error_val_list.append(np.array(val_y[val_y!=val_pred].index))

        print(metrics.classification_report(val_y, val_pred))
        test_pred = model.predict(test_label[features])
        pred += test_pred/5

        model.save_model(lightgbm_model_path+'model_%s.txt'%index)


    oof_ = np.argmax(oof, axis=1)
    all_val_f1 = metrics.f1_score(oof_, y, average='macro')
    print('oof f1', all_val_f1)
    
    pred_ = np.argmax(pred, axis=1)
    sub = test_label[['ship']]
    sub['pred'] = pred_
   
    print(sub['pred'].value_counts(1))
    sub['pred'] = sub['pred'].map(type_map_rev)
    pd.DataFrame({
        "pred1":pred[:,0],
        "pred2":pred[:,1],
        "pred3":pred[:,2]
    }).to_csv('submit/PROB_result_'+str(round(all_val_f1,5))+'_'+time.strftime('%Y-%m-%d-%H-%M-%S')+'.csv', index=None, header=None)
    
    sub.to_csv('result.csv', index=None, header=None)
    feature_importance(models)
    print('result_'+str(round(all_val_f1,5))+'_'+time.strftime('%Y-%m-%d-%H-%M-%S'))
    print('围网 oof f1', metrics.f1_score((oof_==1).astype("int"), (y==1).astype("int")))

def train2(index_list,index_list_val,train_label,features,target):
    
     
    train_label[features].head(5).to_csv("kankan.csv")
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_val_f1 = []

    use_focalloss = 1
    focal_loss = lambda x,y: focal_loss_lgb(x, y, 0.25, 2., 3)
    eval_error = lambda x,y: focal_loss_lgb_eval_error(x, y, 0.25, 2., 3)
    

    learning_rate =config.learning_rate
    params = {
        'n_estimators': 50000,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
    #     "min_data_in_leaf":10,
        'learning_rate': learning_rate, 
    #     "bagging_fraction":0.9,
    #     "feature_fraction":0.9,
    #     "feature_freq":10,
    #     "bagging_freq":10,
    #     "lambda_l1":0.8,
    #     "lambda_l2":0.8,
    #     "metric":lgb_f1_score,
    #     "max_depth":6,
    #     'num_leaves': 6, 
    #     "label_weights_":list(train_label['type'].value_counts(1))*10,
    #     "imbalance":True,
        'num_class': 3,
    #     'num_leaves':60,
        'early_stopping_rounds': 50,
    }
    model_result["use mask"] = 1
    if model_result["use mask"]:
        X = train_label[features].copy()
        y = train_label[target]
    else:
        X = train_label[features].copy()
        y = train_label[target]
    models = []
    
    oof = np.zeros((len(X), 3))
    evals_result = {}  #记录训练结果所用
    error_val_list =[]
    for index in range(5):
        train_idx, val_idx = index_list[str(index+1)].astype(int),index_list_val[str(index+1)].astype(int)
        train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])
    # feval=lgb_f1_score, 
    # fobj=focal_loss, feval=eval_error,
        if use_focalloss ==1:
            model = lgb.train(params, train_set, fobj=focal_loss, feval=eval_error,
                        valid_sets=[train_set, val_set], verbose_eval=300)
        elif use_focalloss ==2:
            model = lgb.train(params, train_set, fobj=focal_loss, feval=lgb_f1_score,
                        valid_sets=[val_set], verbose_eval=300)
        else:
            model = lgb.train(params, train_set,evals_result=evals_result,feval=lgb_f1_score_at_1, 
                        valid_sets=[ val_set], verbose_eval=300)
        models.append(model)
        val_pred = model.predict(X.iloc[val_idx])
        oof[val_idx] = val_pred
        val_y = y.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        val_f1 = metrics.f1_score(val_y, val_pred, average='macro')
        all_val_f1.append(val_f1)
        print(index, 'val f1', val_f1)
        
        error_val_list.append(np.array(val_y[val_y!=val_pred].index))

        print(metrics.classification_report(val_y, val_pred))


        model.save_model(lightgbm_model_path+'model_%s.txt'%index)


    oof_ = np.argmax(oof, axis=1)
    all_val_f1 = metrics.f1_score(oof_, y, average='macro')
    print('oof f1', all_val_f1)

    feature_importance(models)

def submit_file(test_label,features):
    import lightgbm as lgb
    import time
    pred = np.zeros((len(test_label),3))
    for i in os.listdir(lightgbm_model_path):
        model = lgb.Booster(model_file=lightgbm_model_path+i)
        test_pred = model.predict(test_label[features])
        pred += test_pred/5

    type_map_rev = {0: '拖网', 1: '围网', 2: '刺网'}
    pred_ = np.argmax(pred, axis=1)
    sub = test_label[['ship']]
    sub['pred'] = pred_

    print(sub['pred'].value_counts(1))
    sub['pred'] = sub['pred'].map(type_map_rev)
    pd.DataFrame({
        "pred1":pred[:,0],
        "pred2":pred[:,1],
        "pred3":pred[:,2]
    }).to_csv('submit/PROB_result_'+time.strftime('%Y-%m-%d-%H-%M-%S')+'.csv', index=None, header=None)

    sub.to_csv('submit/result_'+time.strftime('%Y-%m-%d-%H-%M-%S')+'.csv', index=None, header=None)
