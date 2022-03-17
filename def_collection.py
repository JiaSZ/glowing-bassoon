
from pyexpat.errors import codes
import numpy as np
import pandas as pd
import os
import xlrd
import math
import matplotlib.pyplot as plt
from simpledbf import Dbf5
from matplotlib.ticker import FuncFormatter

def read_dbf_file(file):
    dbf = Dbf5(file, codec='gbk')
    csv_file = file.replace('.dbf','.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)
    dbf.to_csv(csv_file)
    csv = pd.read_csv(csv_file, encoding = 'gbk')
    df1 = pd.DataFrame(csv)
    return df1

def read_xls_file(file):
    df = pd.read_excel(file)
    return df

def set_pd():
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 80)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
   
def lorenz(list):
    l = len(list)
    list.sort()
    list_agg = []
    for index, val in enumerate(list):
        if index == 0:
            list_agg.append(val)
        if index >= 1:
            m0 = list_agg[index-1] + val
            list_agg.append(m0)
    list_agg.insert(0,0)
    area = 0
    for index, val in enumerate(list_agg):
        if index >= 1:
            m1 = list_agg[index-1]
            m2 = list_agg[index]
            area += (m1 + m2)*1/2
    s = l * list_agg[-1] * 1/2
    gini = 1- area/s
    print('gini = %.3f' % gini)
    x = []
    y = []
    for index, val in enumerate(list_agg):
        x.append(index/( len(list_agg) -1 ))
        y.append(val/list_agg[-1])
    fig = plt.figure(figsize=(8, 6), dpi = 100)
    # plt.style.use('seaborn')
    
    x0 = [0,1]
    y0 = [0,1]

    plt.plot(x0, y0, label = 'Evenness', \
        marker = '+' ,c = 'b', linestyle="-", linewidth=1.5)
    plt.plot(x, y, label = 'lorenz curve', \
        c = 'r', linestyle="-", linewidth=1.5)
    plt.legend(loc = 'upper left')

    def to_percent(temp, position):
        return '%1.0f'%(100*temp) + '%'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.show() 
    
def get_dbf_and_save_csv(file, code):
    dbf = Dbf5(file, codec = code)
    csv_file = file.replace('.dbf','.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)
    dbf.to_csv(csv_file)
    csv = pd.read_csv(csv_file, encoding = code)
    df = pd.DataFrame(csv)
    return df

def connect_df(df1,df2):
    for i1 in df1.index:
        if i1 in df2.index:
            for i2 in df2.columns:
                df1.loc[i1, i2] = df2.loc[i1, i2]
    return df1

def sum_pivot(df, sum_by, sum_data, out_name):
    df_pivot_sum = df[[sum_by, sum_data]]
    df_pivot_sum = df_pivot_sum.groupby(by = sum_by)
    df_pivot_sum = df_pivot_sum[sum_data].agg('sum')
    df_pivot_sum = df_pivot_sum.to_frame()
    df_pivot_sum.columns = [out_name]
    return df_pivot_sum

    
