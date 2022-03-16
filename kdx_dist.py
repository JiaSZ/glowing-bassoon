
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

def get_value(column0):
    list0 = []
    for i in column0:
        list0.append(i.value)
    list0.pop(0)
    return list0

def step_x(x_list, n):
    x = x_list
    x_km = []
    range_x = math.ceil( max(x)/n )
    for i in range(range_x):
        x_km.append( n/2 + i * n)
    return x_km

def aver_y(y, x):
    y_aver = []
    for i in range( x_range ):
        x_index = [i2 for i2,v in enumerate(dist) if (i*n <= v < (i+1)*n)]
        y_aver_c = np.average([y[i2] for i2 in x_index])
        y_aver.append(y_aver_c if math.isnan(y_aver_c) == False else \
            np.average(y_aver[-3:])  )
    return y_aver

def weight_y(y, x):
    y_weight = []
    for i in range( x_range ):
        x_index = [i2 for i2,v in enumerate(dist) if (i*n <= v < (i+1)*n)]
        sum1 = 0
        sum2 = 0
        for i2 in x_index:
            sum1 += df1['P_2020E'][i2]*y[i2]
            sum2 += df1['P_2020E'][i2]
        y_weight.append(sum1/sum2 if sum2 != 0 else 0)
    for index, value in enumerate(y_weight):
        if value == 0:
            y_weight[index] = y_weight[index-1]

    return y_weight



def sum_p(x, p):
    p_sum = []
    for i in range( x_range ):
        x_index = [i2 for i2,v in enumerate(dist) if (i*n <= v < (i+1)*n)]
        p_sum_c = [p[i2] for i2 in x_index]
        p_sum.append( sum(p_sum_c)/10000 if sum(p_sum_c)/10000 else 0) 
    return p_sum

def sum_c(x, c):
    c_sum = []
    for i in range( x_range ):
        x_index = [i2 for i2,v in enumerate(dist) if (i*n <= v < (i+1)*n)]
        c_sum_c = [c[i2] for i2 in x_index]
        c_sum.append( sum(c_sum_c) )
    return c_sum

def scatter0(y):
    Y = df1[y]
    X = dist
    fig = plt.figure(figsize=(8, 6), dpi = 100)
    # plt.style.use('seaborn')

    plt.scatter(X, Y, s = 9, label = y)
    plt.xlabel(str(x) + ' (km)')
    plt.ylabel(y)
    # plt.xlim(0,120)
    plt.ylim(-0.5, 10)
    
    '''趋势线'''
    # z = np.polyfit(X, Y, 1)
    # p = np.poly1d(z)
    # zlist0 = []
    # for i in z:
    #     temp = '%.2e' % i
    #     zlist0.append(temp)
    # label2 = 'Y = a2 * X^2 + a1 * X + b\n' + str(zlist0)
    # mpl.pylab.plot(X,p(X),"r--", label = label2, c = 'r')

    ''' n 公里平均值,[)'''

    y_weight = weight_y(Y, X)

    plt.plot(x_list, y_weight, label = 'weight_by_' + str(n) + 'km', \
        marker = '+' ,c = 'r', linestyle="-", linewidth=1.5)

    plt.legend(loc = 'upper left')
    print(plt.style.available)
    
    plt.show()

def scatter0_add_p(y, p):
    
    Y = df1[y]
    X = dist
    P = df1[p]
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi = 120)
    ax2 = ax1.twinx()
    ax1.scatter(X, Y, s = 9, label = y)
    ax1.set_xlabel(str(x) + ' (km)')
    ax1.set_ylabel(y)
    ax1.set_ylim(-0.5 ,10)
    # ax1.set_xlim(0, 120)

    '''n 公里总人口[)'''

    p_sum = sum_p(X, P)
    ax2.plot(x_list, p_sum, label = 'p_sum (*10,000)', c = 'b',\
        marker = '+', mec = 'k')
    ax2.set_ylim(0,450)
    y_aver = aver_y(Y, X)
    ax1.plot(x_list, y_aver, label = 'weight_by_' + str(n) + 'km', \
        marker = '+', mec='k', c = 'r', linestyle="-", linewidth=1.5)
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    plt.show()

def scatter0_p_and_c(c1, c2, p):
    c1 = df1[c1]
    c2 = df1[c2]
    x = dist
    p = df1[p]

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi = 120)
    ax2 = ax1.twinx()

    c1 = sum_c(x, c1)
    c2 = sum_c(x, c2)

    # ax1.set_xlim(0, 120)
    # ax1.set_ylim(0, 300)
    ax1.plot(x_list, c1, label = 'C_r1')
    ax1.plot(x_list, c2, label = 'C_r2')

    p_sum = sum_p(x, p)
    ax2.plot(x_list, p_sum, label = 'p_sum (*10,000)', c = 'b',\
        marker = '+', mec = 'k')
    # ax2.set_ylim(0, 500)
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    plt.show()

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
    # plt.xlabel(x)
    # plt.ylabel(y)
    # plt.xlim(0,120)
    # plt.ylim(-0.5, 10)


if __name__ == '__main__':


    set_pd()

    '''设置目录'''
    os.getcwd()
    try:
        os.chdir('./kdx距离计算')
    except Exception as r:
        os.getcwd()

    '''读取文件, dbf 或 excel'''
    file = 'kdx_District_dist_0223.xls'
    df1 = read_xls_file(file)

    '''处理绘图x轴, 转换单位'''
    df1['x_list'] = df1['Dist']/1000
    dist = df1['x_list']
    
    

    '''绘图参数设置'''
    #n为步距,即多少公里为一组
    y = 'kdx_S_Df'
    x = 'Dist'
    p = 'P_2020E'
    n = 3

    '''计算全局变量, x坐标点和范围'''
    x_list = step_x(df1['x_list'], n)
    x_range = len(x_list)
    
    '''绘图'''
    scatter0(y)
    scatter0_add_p(y, p)


    '''绘制洛伦兹曲线'''
    list_r012 = list(df1['kdx_r012'])
    list_r012.sort()
    lorenz(list_r012)

    # c1 = 'C_r1'
    # c2 = 'C_r2'
    # scatter0_p_and_c(c1, c2, p)

    



