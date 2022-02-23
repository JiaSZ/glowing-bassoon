from email.policy import default
from simpledbf import Dbf5
import os
import pandas as pd
import numpy as np
import re
import xlwt
import xlwt
import warnings

'''



'''
def set_dir(str):
    ml1 = os.getcwd()
    try:
        os.chdir('./kdx分析计算')
    except Exception as r:
        pass
    print('set dir as \n%s' % ml1)
    return ml1

def set_pd():
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 80)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    warnings.filterwarnings('ignore')

def set_model():
    print('-----------------------Model----------------------')
    print('supply = ', supply, '\nO_type = ', O_type, '\ngo', distance, 'in', time_limit, 'mins')
    print('Leng_minimum = %d' % Leng_min)
    print('---------------------Starting---------------------')

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


'''1.读取OD的dbf文件 → csv → df1 '''
'''2.seperate OD, sort data'''
def read_OD(file):
    print('...1.read OD_dbf and convert to .csv then to df')
    dbf = Dbf5(file, codec='utf-8')
    csv_file = file.replace('.dbf','.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)
    dbf.to_csv(csv_file)
    csv = pd.read_csv(csv_file, usecols = ['Name', 'Total_Leng'])
    df1 = pd.DataFrame(csv)
    # df1
    # df1.head()
    # df1.tail()
    # df1.isna()

    print('......row nums:', df1.shape[0])
    '''分离出O和D点, 并将df1中的str转为int, 准备后续分析'''
    print('...2.seperate OD, sort data')
    
    O_str = 'O_' + O_type
    df1[O_str] = None
    df1['D_Resi'] = None
    # df1
    for index, row in df1.iterrows():
        str0 = row['Name']
        split0 = re.split(r'(\W+)', str0)
        df1.at[index, O_str] = split0[0]
        df1.at[index, 'D_Resi'] = split0[2]
        if (index+1) % 20000 == 0:
            print('......%.1f %%' % (((index+1)/df1.shape[0])*100))
    print('......100 %...completed')
    df1 = df1.drop(['Name'], axis = 1)
    df1 = df1[[O_str,'D_Resi','Total_Leng']]
    df1[O_str] = df1[O_str].apply(int)
    df1['D_Resi'] = df1['D_Resi'].apply(int)
    df1['Total_Leng'] = df1['Total_Leng'].apply(float)
    print('......set Leng as 10 if Leng should <', Leng_min)
    df1.loc[df1['Total_Leng']<Leng_min,'Total_Leng'] = Leng_min
    # df1
    return df1


'''按行政区统计建面, 将分区常住人口划分至建面, 计算该比例'''
def calc_p_estimate(df_Resi, df_318):
    print('...3.calc p_estimate for df_Resi')
    sum1 = sum_pivot(df_Resi, 'region', '建面', 'sum_region_建面')
    sum2 = sum_pivot(df_318, 'region', 'P_2020E', 'sum_p')
    sum3 = connect_df(sum1, sum2)
    sum3.columns = ['sum_Area', 'sum_P']
    sum3['Area/sumP'] = sum3['sum_P']/sum3['sum_Area']
    for i1 in sum3.index:
        df_Resi.loc[df_Resi.region == i1, 'r(A/P)'] = sum3.loc[i1, 'Area/sumP']
    df_Resi['p_estimate'] = df_Resi['建面'] * df_Resi['r(A/P)']
    return df_Resi


'''3.将D居住点的人口、所属区、街道等信息匹配到df1上'''
def connect_Resi(df, df_Resi):
    print('...3.connect df_Resi to df_OD')
    print('......connecting')
    for row in df.itertuples():
        odid = getattr(row, 'D_Resi')
        index = row.Index
        df.at[index, 'p_estimate'] = df_Resi.at[odid, 'p_estimate']
        df.at[index, 'region'] = df_Resi.at[odid, 'region']
        df.at[index, 'District'] = df_Resi.at[odid, 'District']
        if (index+1) % 30000 == 0:
            print('......%.1f %%' % ((index+1)/df.shape[0]*100))
    print('......100 %...completed')
    return df


'''5.在df1上计算时间t, 时间函数f(t),以及需求人口*距离即f*D_p_estimate'''
def calc_fD(df):
    print('...5.calc t_OD, f(t_OD), f*D')
    df['t_OD'] = df['Total_Leng']/distance*time_limit
    df_t_OD = df.t_OD
    df = df.drop('t_OD', axis = 1)
    df.insert(3, 't_OD', df_t_OD)
    
    df['f(t_OD)'] = 1/df['t_OD']
    df['f*D']=df['f(t_OD)']*df['p_estimate']
    print('......completed')
    return df

'''6.基于设施点O; 求和f*D'''
def fd_sum_by_O(df1):

    '''df2记录sum(f*D)_by_O '''
    print('...6.sum(f*D) on Origin')
    print('......summing f*D on Origin')
    df2 = df1.pivot_table(index = 'O_' + O_type, values = 'f*D', aggfunc = 'sum')
    df2.columns = ['sum(f*D)_by_O']
    

    '''将df2传回df1'''
    print('......connecting sum(f*D) to df_OD')
    dict_df2 = {}
    for index, row in df2.iterrows():
        dict_df2[index] = row['sum(f*D)_by_O']

    for index, row in df1.iterrows():
        df1.at[index, 'sum(f*D)_by_O'] = dict_df2[row['O_' + O_type]]
        if (index+1) % 20000 == 0:
            print('......%.1f %%' % ((index+1)/df1.shape[0]*100))
    print('......100 %...completed')
    return df1


'''7.获得供给能力'S'=P/C的字典'''


'''链接供给能力'S'=P/C'''
def connect_S(df):

    print('...7.calc Supply of Origin')
    print('......supply =',supply)

    if supply == 'P/C':
        print('......get P/C form df_318')
        str1 = 'C_' + O_type
        df_318s = df_318[['P_2020E' , str1]]
        df_318s['P/C'] = df_318s['P_2020E']/df_318s[str1]
        
        print('......connecting O_District to df_OD')
        file = O_type + '.dbf'
        df_O = get_dbf_and_save_csv(file, 'gbk')
        df_O = df_O[['odid','District']]
        df_O = df_O.set_index('odid')
        str2 = 'O_' + O_type
        for row in df.itertuples():
            index = row.Index
            O_odid = getattr(row, str2)
            df.at[index, 'O_District'] = df_O.at[O_odid, 'District']
        print('......connecting P/C in df_318s to df_OD')
        for row in df.itertuples():
            df.at[row.Index, 'S=P/C'] = df_318s.at[getattr(row, 'O_District'), 'P/C']
        df['S*f/sum'] = df['S=P/C']*df['f(t_OD)']/df['sum(f*D)_by_O']
        print('......completed')
    elif supply == 'sum(D*f)':
        print('......calculating')
        df['S=sum(D*f)'] = df['sum(f*D)_by_O']
        df['S*f/sum'] = df['S=sum(D*f)']*df['f(t_OD)']/df['sum(f*D)_by_O']
    else:
        str1 = 'S='+str(supply)
        df[str1] = supply
        df['S*f/sum'] = df[str1]*df['f(t_OD)']/df['sum(f*D)_by_O']
    print('......df1 is completed:\n\n', df, '\n')
    return df

'''8.按D汇总Ai'''
def calc_Ai(df):

    '''基于D住宅点, 透视汇总负荷*距离, 形成各住宅点的可达性---df1_kdx'''
    print('...8.sum \'S*f/sum(D*f)\' on Destination')
    print('......summing')
    Ai = df1.groupby(by = 'D_Resi')
    Ai = Ai['S*f/sum'].agg('sum')
    Ai = Ai.to_frame()
    # df1_kdx
    Ai.columns = ['Ai=sum(S*f/sum())']
    print('......completed')
    return Ai

'''对Ai连接人口并加权,连接所属地区及街道'''
def connect_Ai(Ai, df_Resi):

    '''Ai匹配df_Resi信息(人口/所属区、乡镇街道)'''
    print('......connecting dict_Resi')
    for row in Ai.itertuples():
        index = row.Index
        Ai.at[index,'region'] = df_Resi.at[index, 'region']
        Ai.at[index,'District'] = df_Resi.at[index, 'District']
        Ai.at[index,'p_estimate'] = df_Resi.at[index, 'p_estimate']
    print('......completed\n')
    return Ai

'''---------------9.1/9.2 按乡镇街道/行政区汇总p*A-----------------'''



'''---------------10.1/10.2 按乡镇街道/行政区汇总人口-----------------'''
def connect_p_District(kdx_District, df_Resi):
    print('...10.1 summing and connecting sum_P on District')
    df_Resi_sum = sum_pivot(df_Resi,'District','p_estimate','sum_p')
    dict_sumP = {}
    for index, row in df_Resi_sum.iterrows():
        dict_sumP[index] = row['sum_p']
    
    for index, row in kdx_District.iterrows():
        kdx_District.at[index, 'sum_p'] = dict_sumP[index]
    '''注意:人口是按区分到建面的, 因此街道汇总人口将使用建面拟合数, 而不使用供需比所用的推测数'''
    kdx_District['kdx'] = kdx_District['sum_p*A']/kdx_District['sum_p']
    kdx_District = kdx_District.sort_values(by = 'kdx', ascending = False)
    return kdx_District
    
def connect_p_region(kdx_region, df_Resi):
    print('...10.2 summing and connecting sum_P on region')
    df_Resi_region = sum_pivot(df_Resi,'region','p_estimate','sum_p')
    dict_sumP = {}
    for index, row in df_Resi_region.iterrows():
        dict_sumP[index] = row['sum_p']
    for index, row in kdx_region.iterrows():
        kdx_region.at[index, 'sum_p'] = dict_sumP[index]
    kdx_region['kdx'] = kdx_region['sum_p*A']/kdx_region['sum_p']
    kdx_region = kdx_region.sort_values(by = 'kdx', ascending = False)
    return kdx_region

def save():

    print('..11.saving to xls')

    if supply == 'P/C':
        str1 = '_S_P_C'
    elif supply == 'sum(D*f)':
        str1 = '_S_sum_Df'
    else:
        str1 = '_S_' + str(supply)

    file_kdx_region = 'kdx_region_' + O_type + str1 + '.xls'
    kdx_region.columns = ['sum_p_A','sum_p','kdx']
    
    file_kdx_District = 'kdx_District_' + O_type + str1 + '.xls'
    kdx_District.columns = ['sum_p_A','sum_p','kdx']
    
    if supply != 'P/C' and supply != 'sum(D*f)':
        kdx_District['kdx_1w'] = kdx_District['kdx']*10000
        kdx_region['kdx_1w'] = kdx_region['kdx']*10000
    
    kdx_region.to_excel(file_kdx_region)
    kdx_District.to_excel(file_kdx_District)
    
    print('......', file_kdx_District , ':\n', kdx_District, '\n')
    print('......', file_kdx_region , ':\n', kdx_region)
    print('\n---------------------MISSION COMPLETED---------------------')





if __name__ == '__main__':

    '''设置目录，参数'''
    dir1 = set_dir(str = './kdx分析计算')
    set_pd()

    '''模型参数设置'''
    #设施供给, supply = 'P/C', sum(D*f), 或指定1/4.74
    supply = 'P/C'
    O_type = 'r2'
    distance = 300
    time_limit = 5
    Leng_min = 10
    set_model()

    '''文件设置'''
    file_OD = O_type + '_to_house.dbf'
    Resi_file = 'Resi.dbf'
    District_file = 'districts318.dbf'
    

    '''全局参数设置'''
    df1 = read_OD(file_OD)
    df_Resi = get_dbf_and_save_csv(Resi_file, 'gbk')
    df_Resi = df_Resi.set_index('odid')
    df_318 = get_dbf_and_save_csv(District_file, 'gbk')
    df_318 = df_318.set_index('District')


    '''计算住宅点建面人口(按区)并连接至df1'''
    df_Resi = calc_p_estimate(df_Resi, df_318)
    df1 = connect_Resi(df1, df_Resi)
    
    '''计算分母: 单点需求, 按设施点汇总'''
    df1 = calc_fD(df1)
    df1 = fd_sum_by_O(df1)
    
    '''计算设施点供给能力(按模型类型)'''
    '''计算设施点载荷能力, 计算单设施对需求点的可达性贡献'''
    df1 = connect_S(df1)
    
    '''加总可达性贡献, 计算需求点kdx'''
    Ai = calc_Ai(df1)
    # Ai.loc[43]
    # df1.loc[df1.D_Resi == 43]

    '''连接估计人口(权数)'''
    Ai = connect_Ai(Ai, df_Resi)

    '''按计算范围汇总加权kdx'''
    print('..9.1/9.2 summing p*A on District/region')
    kdx_District = sum_pivot(Ai, 'District', 'p_estimate', 'sum_p*A')
    kdx_region = sum_pivot(Ai, 'region', 'p_estimate', 'sum_p*A')
    
    '''加权kdx除以总人口'''
    kdx_District = connect_p_District(kdx_District, df_Resi)
    kdx_region = connect_p_region(kdx_region, df_Resi)

    save()
    




    # Ai.loc[Ai.District == '交道口街道']
    # df1.loc[df1.D_Resi == 16865]
    
    # df1.loc[:,'Total_Leng':'D_p_estimate'] = df1.loc[:,'Total_Leng':'D_p_estimate'].applymap(lambda x:'{:.2f}'.format(x))
    # df1.loc[:,'f(t_OD)':'S=P/C'] = df1.loc[:,'f(t_OD)':'S=P/C'].applymap(lambda x:'{:.2f}'.format(x))
    # # np.mean(list(df_318['C_r2']))
    # df_District = pd.DataFrame(O_csv)
    # df_District = df_District[['地区', 'P_2020E']]
    # df_District
    
    # p_District = {}
    # for index, row in df_District.iterrows():
    #     p_District[row['地区']] = row['P_2020E']
    # df_District
    
    # # sum(df1.loc[df1.O_r2 == 455]['f*D'])
    
    
    # df1
    for index, row in df1.iterrows():
        str0 = row['Name']
        split0 = re.split(r'(\W+)', str0)
        df1.at[index, O_str] = split0[0]
        df1.at[index, 'D_Resi'] = split0[2]
        if (index+1) % 20000 == 0:
            print('......%.1f %%' % (((index+1)/df1.shape[0])*100))
    print('......100 %...completed')
    df1 = df1.drop(['Name'], axis = 1)
    df1 = df1[[O_str,'D_Resi','Total_Leng']]
    df1[O_str] = df1[O_str].apply(int)
    df1['D_Resi'] = df1['D_Resi'].apply(int)
    df1['Total_Leng'] = df1['Total_Leng'].apply(float)
    # df1
    return df1

'''3.获取住宅点D的属性字典'''
def get_dict_and_df_Resi(Resi_file):
    print('...3.get dict_Resi, df_Resi')
    dbf = Dbf5(Resi_file, codec='gbk')
    # dbf.numrec
    Resi_file_csv = Resi_file.replace('.dbf', '.csv')
    if os.path.exists(Resi_file_csv):
        os.remove(Resi_file_csv)
    dbf.to_csv(Resi_file_csv)
    Resi_csv = pd.read_csv(Resi_file_csv, encoding = 'gbk')
    df_Resi = pd.DataFrame(Resi_csv)
    # df2.shape[0]
    dict_Resi = {}
    for index, row in df_Resi.iterrows():
        odid = row['odid']
        dict_Resi[odid] = {}
        #原表中为每百平人数，因此除以100
        dict_Resi[odid]['p_estimate'] = row['p_estimate']
        dict_Resi[odid]['District'] = row['District']
        dict_Resi[odid]['region'] = row['region']
        if (index+1) % 20000 == 0:
            print('......%.1f %%' % ((index+1)/df_Resi.shape[0]*100))
    
    print('......100 %...completed')
    # dict_Resi[1]
    return dict_Resi, df_Resi

'''4.将D居住点的人口、所属区、街道等信息匹配到df1上'''
def connect_Resi(df, dict_Resi):

    print('...4.connect dict_Resi to df_OD')

    df['D_p_estimate'] = None
    df['region'] = None
    df['District'] = None
    # df1.at[0, 'D_p_estimate'] = dict_Resi[48396]['p_estimate']
    # df1.at[0, 'region'] = dict_Resi[48396]['region']
    # df1.at[0, 'District'] = dict_Resi[48396]['District']
    for index, row in df.iterrows():
        odid = row['D_Resi']
        df.at[index, 'D_p_estimate'] = dict_Resi[odid]['p_estimate']
        df.at[index, 'region'] = dict_Resi[odid]['region']
        df.at[index, 'District'] = dict_Resi[odid]['District']
        if (index+1) % 20000 == 0:
            print('......%.1f %%' % ((index+1)/df.shape[0]*100))
    print('......100 %...completed')
    return df


'''5.在df1上计算时间t, 时间函数f(t),以及需求人口*距离即f*D_p_estimate'''
def calc_fD(df):
    print('...5.calc t_OD, f(t_OD), f*D')
    df['t_OD'] = df['Total_Leng']/distance*time_limit
    # df = df[['O_' + O_type,'D_Resi','Total_Leng','t_OD','D_p_estimate','region','District']]
    df_t_OD = df.t_OD
    df = df.drop('t_OD', axis = 1)
    df.insert(3, 't_OD', df_t_OD)
    
    df['f(t_OD)'] = 1/df['t_OD']
    df['f*D']=df['f(t_OD)']*df['D_p_estimate']
    print('......completed')
    return df

'''6.基于设施点O; 求和f*D'''
def fd_sum_by_O(df):

    '''df2记录sum(f*D)_by_O '''
    print('...6.sum(f*D) on Origin')
    print('......summing f*D on Origin')
    df2 = df.pivot_table(index = 'O_' + O_type, values = 'f*D', aggfunc = 'sum')
    df2.columns = ['sum(f*D)_by_O']
    # df1_sum_by_O.shape[0]
    # df1_sum_by_O
    # df2['S/sum(f*D)'] = supply/df2['sum(f*D)']         #
    # df1_sum_by_O
    # len(set(np.array(df[O_str]).tolist()))
    # type(df1_sum_by_O['S/sum(f*D)'][18])

    '''将df2传回df1'''
    print('......connecting sum(f*D) to df_OD')
    dict_df2 = {}
    for index, row in df2.iterrows():
        dict_df2[index] = row['sum(f*D)_by_O']
    # [i for i in dict_O_bear.keys()][:5]
    # dict_O_bear[77]
    # dict_O_bear
    # df['S/sum(f*D)'] = None
    for index, row in df.iterrows():
        df.at[index, 'sum(f*D)_by_O'] = dict_df2[row['O_' + O_type]]
        if (index+1) % 20000 == 0:
            print('......%.1f %%' % ((index+1)/df1.shape[0]*100))
    print('......100 %...completed')
    return df
    # df1
    # df1['S/sum(f*D)'][205757]
    # df1['S*f/sum(D*f)'] = df1['S/sum(f*D)']*df1['f(t_OD)']
    # df1
    # df1['S*f/sum(D*f)'][205756]

'''7.获得供给能力'S'=P/C的字典'''
def get_district_O_supply(file):
    
    print('...7.calc Supply of Origin')
    print('......supply =',supply)
    dict_318_S = {}
    if supply == 'P/C':
        print('......get P/C dict from district_318 file')
        dbf = Dbf5(file, codec='gbk')
        file_csv = file.replace('.dbf', '.csv')
        if os.path.exists(file_csv):
            os.remove(file_csv)
        dbf.to_csv(file_csv)
        df_318 = pd.read_csv(file_csv, encoding = 'gbk')
        df_318 = df_318[['地区', 'Name', 'P_2020E', 'C_' + O_type]]
        df_318.columns = ['district', 'region', 'P', 'C_' + O_type]
        df_318['P/C_' + O_type + '(\'S\')'] = df_318['P']/df_318['C_' + O_type]
        for index, row in df_318.iterrows():
            dict_318_S[row['district']] = row['P/C_' + O_type + '(\'S\')']
    return dict_318_S

'''链接供给能力'S'=P/C'''
def connect_S(df, dict):

    if supply == 'P/C':
        print('......connecting P/C dict to df1 as S')
        for index, row in df.iterrows():
            df.at[index, 'S=P/C'] = dict[row['District']]
        df['S*f/sum'] = df['S=P/C']*df['f(t_OD)']/df['sum(f*D)_by_O']
    elif supply == 'sum(D*f)':
        print('......calculating')
        df['S=sum(D*f)'] = df['sum(f*D)_by_O']
        df['S*f/sum'] = df['S=sum(D*f)']*df['f(t_OD)']/df['sum(f*D)_by_O']
    else:
        str1 = 'S='+str(supply)
        df[str1] = supply
        df['S*f/sum'] = df[str1]*df['f(t_OD)']/df['sum(f*D)_by_O']
    print('......df1 is completed:\n\n', df1, '\n')
    return df

'''8.按D汇总Ai'''
def calc_Ai(df):

    '''基于D住宅点, 透视汇总负荷*距离, 形成各住宅点的可达性---df1_kdx'''
    print('...8.sum \'S*f/sum(D*f)\' on Destination')
    print('......summing')
    Ai = df1.groupby(by = 'D_Resi')
    Ai = Ai['S*f/sum'].agg('sum')
    Ai = Ai.to_frame()
    # df1_kdx
    Ai.columns = ['Ai=sum(S*f/sum())']
    return Ai

'''对Ai连接人口并加权,连接所属地区及街道'''
def connect_Ai(Ai, dict_Resi):

    '''Ai匹配dict_Resi信息(人口/所属区、乡镇街道)'''
    # df1_kdx.shape[0]
    print('......connecting dict_Resi to Ai_point...')
    for index, row in Ai.iterrows():
        Ai.at[index,'region'] = dict_Resi[index]['region']
        Ai.at[index,'District'] = dict_Resi[index]['District']
        Ai.at[index,'p_estimate'] = dict_Resi[index]['p_estimate']
    print('......100 %...completed')
    Ai['p*A'] = Ai['Ai=sum(S*f/sum())']*Ai['p_estimate']
    # df1_kdx
    # type(df1_kdx['p*A'][1])
    return Ai

'''---------------9.1/10.1 按乡镇街道/行政区汇总p*A-----------------'''
def sum_pA_District(Ai):
    
    print('...9.1 summing p*A on District')
    kdx_District = Ai.groupby(by = 'District')
    kdx_District = kdx_District['p*A'].agg('sum')
    kdx_District = kdx_District.to_frame()
    kdx_District.columns = ['sum_p*A']
    # kdx_District
    return kdx_District

def sum_pA_region(Ai):

    print('..10.1 summing p*A on region')
    kdx_region = Ai.groupby(by = 'region')
    kdx_region = kdx_region['p*A'].agg('sum')
    kdx_region = kdx_region.to_frame()
    kdx_region.columns = ['sum_p*A']
    # kdx_region
    return kdx_region

'''---------------9.2/10.2 按乡镇街道/行政区汇总人口-----------------'''
def connect_p_District(kdx_District, df_Resi):
    print('...9.2 summing and connecting sum_P on District')
    df_Resi = df_Resi[['District', 'p_estimate']]
    df_Resi = df_Resi.groupby(by = 'District')
    df_Resi = df_Resi['p_estimate'].agg('sum')
    df_Resi = df_Resi.to_frame()
    df_Resi.columns = ['sum_p']
    dict_sumP = {}
    for index, row in df_Resi.iterrows():
        dict_sumP[index] = row['sum_p']
    
    for index, row in kdx_District.iterrows():
        kdx_District.at[index, 'sum_p'] = dict_sumP[index]
    '''注意:人口是按区分到建面的, 因此街道汇总人口将使用建面拟合数, 而不使用供需比所用的推测数'''
    kdx_District['kdx'] = kdx_District['sum_p*A']/kdx_District['sum_p']
    kdx_District = kdx_District.sort_values(by = 'kdx', ascending = False)
    return kdx_District
    
def connect_p_region(kdx_region, df_Resi):
    print('..10.2 summing and connecting sum_P on region')
    df_Resi = df_Resi[['region', 'p_estimate']]
    df_Resi = df_Resi.groupby(by = 'region')
    df_Resi = df_Resi['p_estimate'].agg('sum')
    df_Resi = df_Resi.to_frame()
    df_Resi.columns = ['sum_p']
    dict_sumP = {}
    for index, row in df_Resi.iterrows():
        dict_sumP[index] = row['sum_p']
    
    for index, row in kdx_region.iterrows():
        kdx_region.at[index, 'sum_p'] = dict_sumP[index]
    kdx_region['kdx'] = kdx_region['sum_p*A']/kdx_region['sum_p']
    kdx_region = kdx_region.sort_values(by = 'kdx', ascending = False)
    return kdx_region

def save():

    print('..11.saving to xls')

    if supply == 'P/C':
        str1 = '_S_P_C'
    elif supply == 'sum(D*f)':
        str1 = '_S_sum_Df'
    else:
        str1 = '_S_' + str(supply)

    file_kdx_region = 'kdx_region_' + O_type + str1 + '.xls'
    kdx_region.columns = ['sum_p_A','sum_p','kdx']
    
    file_kdx_District = 'kdx_District_' + O_type + str1 + '.xls'
    kdx_District.columns = ['sum_p_A','sum_p','kdx']
    
    if supply != 'P/C' and supply != 'sum(D*f)':
        kdx_District['kdx_1w'] = kdx_District['kdx']*10000
        kdx_region['kdx_1w'] = kdx_region['kdx']*10000
    
    kdx_region.to_excel(file_kdx_region)
    kdx_District.to_excel(file_kdx_District)
    
    print('......', file_kdx_District , ':\n', kdx_District, '\n')
    print('......', file_kdx_region , ':\n', kdx_region)
    print('\n---------------------MISSION COMPLETED---------------------')


    # kdx_District
    # df1.loc[df1.District == '大栅栏街道']
    # df1.loc[df1.District == '椿树街道']
    # df1_sum_by_O.loc[1434:1435]
    # df1_kdx.loc[60013:60014]
    # df1_kdx.loc[4552:4558]

    # 0.994272+0.005728+0.083173+0.691598+0.104239+0.12099

    # kdx_District
    # kdx_District.loc['椿树街道']



if __name__ == '__main__':

    '''设置目录，参数'''
    os.getcwd()
    try:
        os.chdir('./kdx分析计算')
    except Exception as r:
        print(r)

    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 80)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    warnings.filterwarnings('ignore')


    '''参数设置'''
    '''设施供给能力模式, 可选'P/C', sum(D*f), 或直接指定1/4.74'''
    # supply = 1            # 1 or 4.74  #'P/C', 'sum(D*f)'
    supply = 1
    O_type = 'r2'
    distance = 300
    time_limit = 5
    print('---------------------Setting---------------------')
    print('supply = ', supply, '\nO_type = ', O_type, '\ngo', distance, 'in', time_limit, 'mins')
    print('---------------------Starting---------------------')
    '''计算'''
    file_OD = O_type + '_to_house.dbf'
    df1 = read_OD(file_OD)
    
    
    Resi_file = 'Resi.dbf'
    Resi_data = get_dict_and_df_Resi(Resi_file)
    dict_Resi = Resi_data[0]
    df_Resi = Resi_data[1]
    # dict_Resi[1]
    
    df1 = connect_Resi(df1, dict_Resi)
    
    df1 = calc_fD(df1)
    
    df1 = fd_sum_by_O(df1)
    
    District_file = 'districts318.dbf'
    dict_318_S = get_district_O_supply(District_file)
    
    
    df1 = connect_S(df1, dict_318_S)
    
    Ai = calc_Ai(df1)
    # Ai.loc[43]
    # df1.loc[df1.D_Resi == 43]
    Ai = connect_Ai(Ai, dict_Resi)
    
    
    kdx_District = sum_pA_District(Ai)
    kdx_region = sum_pA_region(Ai)


    kdx_District = connect_p_District(kdx_District, df_Resi)
    kdx_region = connect_p_region(kdx_region, df_Resi)
    
    save()
    


    # Ai.loc[Ai.District == '交道口街道']
    # df1.loc[df1.D_Resi == 16865]


    


    
    
    # df1.loc[:,'Total_Leng':'D_p_estimate'] = df1.loc[:,'Total_Leng':'D_p_estimate'].applymap(lambda x:'{:.2f}'.format(x))
    # df1.loc[:,'f(t_OD)':'S=P/C'] = df1.loc[:,'f(t_OD)':'S=P/C'].applymap(lambda x:'{:.2f}'.format(x))
    # # np.mean(list(df_318['C_r2']))
    # df_District = pd.DataFrame(O_csv)
    # df_District = df_District[['地区', 'P_2020E']]
    # df_District
    



    # p_District = {}
    # for index, row in df_District.iterrows():
    #     p_District[row['地区']] = row['P_2020E']
    # df_District
    
    # # sum(df1.loc[df1.O_r2 == 455]['f*D'])
    
    
