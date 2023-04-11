import os 
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.datasets import gbsg
from BandDeepHitSingle import BandedDeepHitSingle
from pycox.evaluation import EvalSurv
from sklearn.model_selection import KFold

def BDHS_GBSG():
    #设置种子
    se = random.randint(0,9999)
    np.random.seed(se)
    _ = torch.manual_seed(se)
    
    # 保存C-index目录
    save_C_index = os.path.join('result.csv')
    
    # 导入数据
    dir_path = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), "../..")))
    path = os.path.abspath(os.path.join(dir_path, 'gbsg.csv'))
    df = pd.read_csv(path, index_col=False)
    ci = 0
    
    # 交叉验证
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(df):
        
        df_train = []
        df_val = []
        df_test = []
        
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)
    
        #处理数据
        cols_standardize = ['x3', 'x5', 'x6']
        cols_leave = ['x0', 'x1', 'x2', 'x4']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]

        x_mapper = DataFrameMapper(standardize + leave)

        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_val = x_mapper.transform(df_val).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')

        num_durations = 10
        #labtrans = DeepHitSingle.label_transform(num_durations)
        labtrans = BandedDeepHitSingle.label_transform(num_durations)
        get_target = lambda df: (df['duration'].values, df['event'].values)
        y_train = labtrans.fit_transform(*get_target(df_train))
        y_val = labtrans.transform(*get_target(df_val))

        train = (x_train, y_train)
        val = (x_val, y_val)
        # We don't need to transform the test labels
        durations_test, events_test = get_target(df_test)

        in_features = x_train.shape[1]
        num_nodes = [32, 32]
        out_features = labtrans.out_features
        batch_norm = True
        dropout = 0.1

        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
        model = BandedDeepHitSingle(net, tt.optim.Adam, alpha=0.6, sigma=0.4, duration_index=labtrans.cuts)
        batch_size = 256
        lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=3)
        model.optimizer.set_lr(lr_finder.get_best_lr())

        epochs = 256
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = False
        log = model.fit(x_train, y_train, batch_size, epochs, callbacks,verbose,val_data=val)

        #训练结果
        #surv = model.predict_surv_df(x_test)
        surv = model.interpolate(10).predict_surv_df(x_test)
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        ci += ev.concordance_td('antolini')


    ci /= k
    #输出
    if not os.path.exists(save_C_index):
        with open(save_C_index, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["seed", "c-index"])
            csv_writer.writerow([se, ci])
        f.close()
    else:
        with open(save_C_index, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([se, ci])
        f.close()