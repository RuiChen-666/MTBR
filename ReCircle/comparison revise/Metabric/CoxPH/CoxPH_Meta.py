import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt
import os
import csv
import random
import pandas as pd

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.model_selection import KFold

def CoxPH_Meta():
    #设置种子
    se = random.randint(0,9999)
    np.random.seed(se)
    _ = torch.manual_seed(se)
    
    # 保存C-index目录
    save_C_index = os.path.join('result.csv')
    
    # 导入数据
    dir_path = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), "..")))
    path = os.path.abspath(os.path.join(dir_path, 'metabric.csv'))
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
    
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]

        x_mapper = DataFrameMapper(standardize + leave)

        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_val = x_mapper.transform(df_val).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')


        get_target = lambda df: (df['duration'].values, df['event'].values)
        y_train = get_target(df_train)
        y_val = get_target(df_val)
        durations_test, events_test = get_target(df_test)
        val = x_val, y_val

        in_features = x_train.shape[1]
        num_nodes = [32, 32]
        out_features = 1
        batch_norm = True
        dropout = 0.1
        output_bias = False

        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)

        model = CoxPH(net, tt.optim.Adam)

        batch_size = 256
        lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)

        model.optimizer.set_lr(lrfinder.get_best_lr())

        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = False

        log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                        val_data=val, val_batch_size=batch_size)

        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(x_test)

        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        ci += ev.concordance_td()
        # time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
        # ibs = ev.integrated_brier_score(time_grid)
        # nbll = ev.integrated_nbll(time_grid)
    
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
    