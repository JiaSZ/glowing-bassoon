from pyexpat.errors import codes
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from simpledbf import Dbf5
from matplotlib.ticker import FuncFormatter






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
        marker = '+' ,c = 'r', linestyle="-", linewidth=1.5)
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


x = [1,2,3,3,3,2]
lorenz(x)
