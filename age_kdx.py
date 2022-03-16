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

def set_pd():
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 80)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    warnings.filterwarnings('ignore')

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
    df1 = get_dbf_and_save_csv(file, 'utf-8')
    df1 = df1[['Name', 'Total_Leng']]
    
    print('......row nums:', df1.shape[0])
    '''分离出O和D点, 并将df1中的str转为int, 准备后续分析'''
    print('...2.seperate OD, sort data')
    
    O_str = 'O_' + O_type
    df1[O_str] = None
    df1['D_Resi'] = None
    # df1
    for row in df1.itertuples():
        str0 = getattr(row, 'Name')
        index = row.Index
        split0 = re.split(r'(\W+)', str0)
        df1.at[index, O_str] = split0[0]
        df1.at[index, 'D_Resi'] = split0[2]
        if (index+1) % 30000 == 0:
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
def calc_p_estimate():
    print('...3.calc p_estimate for df_Resi')
    sum1 = sum_pivot(df_Resi, 'region', '建面', 'sum_region_建面')
    sum2 = sum_pivot(df_318, 'region', 'P_2020E', 'sum_p')
    sum3 = connect_df(sum1, sum2)
    sum3.columns = ['sum_Area', 'sum_P']
    sum3['Area/sumP'] = sum3['sum_P']/sum3['sum_Area']
    
    for i1 in sum3.index:
        df_Resi.loc[df_Resi.region == i1, 'r(A/P)'] = sum3.loc[i1, 'Area/sumP']
    df_Resi['p_estimate'] = df_Resi['建面'] * df_Resi['r(A/P)']
    return

'''3.获取住宅点D的属性字典'''
'''4.将D居住点的人口、所属区、街道等信息匹配到df1上'''
def connect_Resi(df, df_Resi):
    print('...4.get dict and connect df_Resi to df_OD')
    print('......get dict of df_Resi')
    dict_Resi = {}
    for row in df_Resi.itertuples():
        region, District = getattr(row, 'region'), getattr(row, 'District')
        p = getattr(row, 'p_estimate')
        index = row.Index
        dict_Resi[index] = {'region': region, 'District': District, 'p_estimate': p}
        if (index+1) % 40000 == 0:
            print('......%.1f %%' % ((index+1)/df_Resi.shape[0]*100))
    print('......100 %...completed')
    # df1.at[0, 'p_estimate'] = dict_Resi[48396]['p_estimate']
    # df1.at[0, 'region'] = dict_Resi[48396]['region']
    # df1.at[0, 'District'] = dict_Resi[48396]['District']
    print('......connecting')
    for row in df.itertuples():
        odid = getattr(row, 'D_Resi')
        index = row.Index
        df.at[index, 'p_estimate'] = dict_Resi[odid]['p_estimate']
        df.at[index, 'region'] = dict_Resi[odid]['region']
        df.at[index, 'District'] = dict_Resi[odid]['District']
        # row['p_estimate'] = dict_Resi[odid]['p_estimate']
        # row['region'] = dict_Resi[odid]['region']
        # row['District'] = dict_Resi[odid]['District']
        if (index+1) % 40000 == 0:
            print('......%.1f %%' % ((index+1)/df.shape[0]*100))
    print('......100 %...completed')
    return df

def calc_age_group(df, df_age):
    print('......connecting age_info to df_OD')
    for row in df.itertuples():
        index = row.Index
        df.at[index, 'r_65plus'] = df_age.at[ getattr(row,'District'), 'r_65plus']
        # df.at[index, 'r_14below'] = df_age.loc[ val['District'] ]['y_14below']
        if (index+1) % 30000 == 0:
            print('......%.1f %%' % ((index+1)/df.shape[0]*100))
    if str_p == 'p_65plus':
        df[str_p] = df['p_estimate']*df['r_65plus']
    elif str_p == 'p_65below':
        df[str_p] = df['p_estimate'] * (1-df['r_65plus'])
    print('......100 %...completed')
    return df


'''5.在df1上计算时间t, 时间函数f(t),以及需求人口*距离即f*p_estimate'''
def calc_fD(df, str_p):
    print('...5.calc t_OD, f(t_OD), f*D')
    # print('......dropping Leng >', distance)
    # df = df.drop( df[df.Total_Leng > distance].index )
    df['t_OD'] = df['Total_Leng']/distance*time_limit
    # df = df[['O_' + O_type,'D_Resi','Total_Leng','t_OD','p_estimate','region','District']]
    df_t_OD = df.t_OD
    df = df.drop('t_OD', axis = 1)
    df.insert(3, 't_OD', df_t_OD)
    df['f(t_OD)'] = 1/df['t_OD']
    '''无论是计算哪类人群, 总需求的计算公式一致, 都是老年人这个群体乘需求密度'''
    if str_p == 'p_65plus' or str_p == 'p_65below':
        print('......get f*D carefully when p_type =', str_p)
        df.loc[df['Total_Leng']<=distance, 'f*D'] = df.loc[df['Total_Leng']<=distance,'f(t_OD)'] * \
            df.loc[df['Total_Leng']<=distance, 'p_estimate'] * \
            ( D_intensity * df.loc[df['Total_Leng']<=distance, 'r_65plus'] + \
            (1 - df.loc[df['Total_Leng']<=distance, 'r_65plus']) )
        df.loc[df['Total_Leng']>distance, 'f*D'] = (1 - df.loc[df['Total_Leng']>distance, 'r_65plus'])*\
                df.loc[df['Total_Leng']>distance, 'p_estimate']*\
                df.loc[df['Total_Leng']>distance, 'f(t_OD)']
        # for index, val in df.iterrows():
        #     df.loc[index, 'f*D'] = ( val['f(t_OD)'] * val['p_estimate'] * ( D_intensity * val['r_65plus'] +\
        #         (1 - val['r_65plus']) )) if (val['Total_Leng'] <= distance) else ( (1 - val['r_65plus'])*\
        #         val['p_estimate']*val['f(t_OD)'] )
        #     if (index+1) % 15000 == 0:
        #         print('......%.1f %%' % ((index+1)/df.shape[0]*100))
        # print('......100 %...completed')
    elif str_p == 'p_estimate':
        print('......get f*D when p_type =', str_p, ', this will be quick')
        df['f*D'] = df['f(t_OD)'] * df[str_p]
    print('......completed')
    return df

'''6.基于设施点O; 求和f*D'''
def fd_sum_by_O(df):

    '''df2记录sum(f*D)_by_O '''
    print('...6.sum(f*D) on Origin')
    print('......summing sum(f*D)_by_O')

    df2 = df.pivot_table(index = 'O_' + O_type, values = 'f*D', aggfunc = 'sum')
    df2.columns = ['sum(f*D)_by_O']

    '''将df2传回df1'''
    print('......connecting sum(f*D) to df_OD')
    dict_df2 = {}
    for index, row in df2.iterrows():
        dict_df2[index] = row['sum(f*D)_by_O']
    for row in df.itertuples():
        index = row.Index
        str1 = 'O_' + O_type
        O_index = getattr(row, str1)
        df.at[index, 'sum(f*D)_by_O'] = dict_df2[O_index]
        if (index+1) % 40000 == 0:
            print('......%.1f %%' % ((index+1)/df1.shape[0]*100))
    print('......100 %...completed')
    return df



'''7.获得供给能力'S'=P/C的字典并链接到df'''

def connect_S(df):
    print('...7.calc Supply of Origin')
    print('......supply =',supply)

    if supply == 'P/C':
        print('......get P/C form df_318')
        str1 = 'C_' + O_type
        df_318s = df_318[['District', 'P_2020E' , str1]]
        df_318s['P/C'] = df_318s['P_2020E']/df_318s[str1]
        df_318s = df_318s.set_index('District')
        
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
    print('......df1 is completed:\n\n', df.columns, '\n')
    return df

'''8.按D汇总Ai'''
def calc_Ai(df):

    '''基于D住宅点, 透视汇总负荷*距离, 形成各住宅点的可达性---df1_kdx'''
    '''仅高龄人群会舍去较远点'''
    print('...8.sum \'S*f/sum(D*f)\' on Destination as Ai, connect df_Resi info')
    Ai = sum_pivot(df, 'D_Resi', 'S*f/sum', 'Ai=sum(S*f/sum())')
    l1 = len(Ai)
    if str_p == 'p_65plus':
        df = df.drop( df[df.Total_Leng > distance].index )
        Ai = sum_pivot(df, 'D_Resi', 'S*f/sum', 'Ai=sum(S*f/sum())')
        l2 = len(Ai)
        print('......Since sstr_p is p_65plus, l(Ai) dropped from %d to %d' % (l1,l2))
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

def connect_age(Ai, df_age):
    print('......connecting age_info')
    for index, row in df_age.iterrows():
        Ai.loc[Ai['District'] == index, str_p.replace('p_','r_')] = row[str_p.replace('p_','r_')]
    Ai[str_p] = Ai[str_p.replace('p_','r_')]*Ai['p_estimate']
    str_pA = str_p + '*A'
    Ai[str_pA] = Ai['Ai=sum(S*f/sum())']*Ai['p_estimate']*Ai[str_p]
    print('......completed')
    return Ai



'''---------------9.1/10.1 按乡镇街道/行政区汇总p*A-----------------'''
def sum_pA_District(Ai, str_p):
    
    print('...9.1 summing p*A on District')
    str_pA = str_p + '*A'
    str_1 = 'sum_(' + str_p + '*A)'
    kdx_District = sum_pivot(Ai, 'District', str_pA, str_1)

    # kdx_District
    return kdx_District

def sum_pA_region(Ai, str_p):

    print('...9.2 summing p*A on region')
    kdx_region = Ai.groupby(by = 'region')
    str_pA = str_p + '*A'
    kdx_region = kdx_region[str_pA].agg('sum')
    kdx_region = kdx_region.to_frame()
    str_1 = 'sum_(' + str_p + '*A)'
    kdx_region.columns = [str_1]
    # kdx_region
    return kdx_region

'''---------------9.2/10.2 按乡镇街道/行政区汇总人口-----------------'''
def connect_p_District(kdx_District, df_Resi):
    print('...10.1 summing and connecting sum_P on District')
    
    df_Resi_sum = sum_pivot(df_Resi,'District','p_estimate','sum_p')
    print('......connecting age to df_district_sum_p')
    for index, row in df_Resi_sum.iterrows():
        df_Resi_sum.at[index, str_p.replace('p_','r_')] = df_age.loc[index, str_p.replace('p_','r_')]
    str1 = 'sum_' + str_p
    df_Resi_sum[str1] = df_Resi_sum['sum_p']*df_Resi_sum[str_p.replace('p_','r_')]
    for index, row in kdx_District.iterrows():
        kdx_District.at[index, str1] = df_Resi_sum.loc[index, str1]
    '''注意:人口是(按)区分到建面的, 因此街道汇总人口将使用建面拟合数, 而不使用供需比所用的推测数'''
    str3 = 'sum_(' + str_p + '*A)'
    kdx_District['kdx'] = kdx_District[str3]/kdx_District[str1]
    kdx_District = kdx_District.sort_values(by = 'kdx', ascending = False)
    return kdx_District


def connect_p_region(kdx_region, df_Resi):
    print('...10.2 summing and connecting sum_P on region')
    
    print('......summing %s from districts' % str_p)
    ''''通过df_Resi获得街道和行政区的人口总和'''
    df_Resi_region = sum_pivot(df_Resi,'region','p_estimate','sum_p')
    df_Resi_jd = sum_pivot(df_Resi,'District','p_estimate','sum_p')

    ''''街道和行政区的对应关系'''
    df_318_l = df_318[['region','District']]
    df_318_l = df_318_l.set_index(['District'])
    
    ''''街道人口总和, 增加对应区人口和(老年化)率'''
    str_r = str_p.replace('p_','r_')
    for index, row in df_Resi_jd.iterrows():
        df_Resi_jd.at[index, 'region'] = df_318_l.loc[index, 'region']
        df_Resi_jd.at[index, str_r] = df_age.loc[index, str_r]
    
    '''计算各街道(老年)人口'''
    if str_p == 'p_65plus':
        df_Resi_jd[str_p] = df_Resi_jd['sum_p']*df_Resi_jd[str_r]
    elif str_p == 'p_65below':
        df_Resi_jd[str_p] = df_Resi_jd['sum_p']*(1-df_Resi_jd[str_r])
    
    '''各街道(老年)人口按区汇总'''
    str1 = 'sum_' + str_p
    df_sum1 = sum_pivot(df_Resi_jd,'region',str_p,str1)
    
    '''在kdx_region中增加区总(老年)人口'''
    for index, row in kdx_region.iterrows():
        kdx_region.at[index, str1] = df_sum1.loc[index][str1]


    str3 = str_1 = 'sum_(' + str_p + '*A)'
    kdx_region['kdx'] = kdx_region[str3]/kdx_region[str1]
    kdx_region = kdx_region.sort_values(by = 'kdx', ascending = False)

    return kdx_region



def save():

    print('..11.saving to xls')

    if supply == 'P/C':
        str1 = '_S_P_C' + '_'
    elif supply == 'sum(D*f)':
        str1 = '_S_sum_Df' + '_'
    else:
        str1 = '_S_' + str(supply) + '_'

    file_kdx_region = 'kdx_region_' + O_type + str1 + str_p + '_Di_' + str(D_intensity) + '.xls'
    # kdx_region.columns = ['sum_p_A','sum_p','kdx']
    
    file_kdx_District = 'kdx_District_' + O_type + str1 + str_p + '_Di_' + str(D_intensity) + '.xls'
    # kdx_District.columns = ['sum_p_A','sum_p','kdx']
    
    str2 = 'kdx_' + str_p
    if supply != 'P/C' and supply != 'sum(D*f)':
        kdx_District['kdx_1w'] = kdx_District[str2]*10000
        kdx_region['kdx_1w'] = kdx_region[str2]*10000
    
    kdx_region.to_excel(file_kdx_region)
    kdx_District.to_excel(file_kdx_District)
    
    print('......', file_kdx_District , ':\n', kdx_District, '\n')
    print('......', file_kdx_region , ':\n', kdx_region)
    print('\n---------------------MISSION COMPLETED---------------------')

    return True

def save_add_column():

    print('..11.saving: adding column to xls')
    #kdxr10_p65plus_sumDf_Di1
    if supply == 'P/C':
        str1 = '_PC_'
    elif supply == 'sum(D*f)':
        str1 = '_sumDf_'
    else:
        str1 = '_' + str(supply).replace('.','') + '_'

    kdx_col_name = 'kdx' + O_type + '_' + str_p.replace('_','') + str1 + 'Di' + str(D_intensity)

    
    df_D = pd.read_excel('kdx_District.xls').set_index('District')
    df_kdx_D = pd.DataFrame(kdx_District['kdx'])
    df_kdx_D.columns = [kdx_col_name]
    df_D = connect_df(df_D, df_kdx_D).fillna(0.0000001)
    df_D.to_excel('kdx_District.xls')
    print('......District Result:\n', df_kdx_D, '\n')


    df_r = pd.read_excel('kdx_region.xls').set_index('region')
    df_kdx_r = pd.DataFrame(kdx_region['kdx'])
    df_kdx_r.columns = [kdx_col_name]
    df_r = connect_df(df_r, df_kdx_r)
    df_r.to_excel('kdx_region.xls')
    print('......region Result:\n', df_kdx_r, '\n')
    print('\n---------------------MISSION COMPLETED---------------------')

    return True




if __name__ == '__main__':

    '''设置目录，参数'''
    os.getcwd()
    try:
        os.chdir('./kdx_年龄计算')
    except Exception as r:
        print(r)

    set_pd()

    '''参数设置'''
    '''设施供给能力模式, 可选'P/C', 'sum(D*f)', 或直接指定1/4.74'''
    '''人群类型: 计算p_65plus or p_65below'''
    # supply = 1            # 1 or 4.74  #'P/C', 'sum(D*f)'
    supply = 1
    O_type = 'r2'
    distance = 240      # 800 or 240 !!
    time_limit = 5
    str_p = 'p_65below'
    D_intensity = 1
    Leng_min = 10
    print('---------------------Setting---------------------')
    print('supply_type = %s\nO_type = %s\np_type = %s, D_intensity = %.1f'\
        % (supply, O_type, str_p, D_intensity))
    print('Speed(for p_65plus): go %d in %d mins' % (distance, time_limit))
    print('Leng_min =', Leng_min)
    print('---------------------Starting---------------------')
    '''计算'''
    file_OD = O_type + '_to_house.dbf'
    age_file = 'kdx_age.dbf'
    District_file = 'districts318.dbf'
    Resi_file = 'Resi.dbf'

    df1 = read_OD(file_OD)
    # df1['D_Resi'].max()
    # df1.shape[0]

    df_Resi = get_dbf_and_save_csv(Resi_file, 'gbk')
    df_Resi = df_Resi.set_index('odid')
    df_318 = get_dbf_and_save_csv(District_file, 'gbk')
    calc_p_estimate()
    # sum(df_Resi['p_estimate'])    
    # sum1 = sum_pivot(df_Resi, 'District' ,'p_estimate','sum_p')
    # sum1
    # sum(sum1['sum_p'])
    # df_Resi

    
    df1 = connect_Resi(df1, df_Resi)
    df_age = get_dbf_and_save_csv(age_file, 'gbk')
    df_age = df_age[['District','r_65plus','r_65below']]
    df_age = df_age.set_index('District')
    
        
    df1 = calc_age_group(df1, df_age)
    df1 = calc_fD(df1, str_p)
    df1 = fd_sum_by_O(df1)
    
    '''获取供给能力, 在P/C条件下进行计算'''
    df1 = connect_S(df1)
    # df1.sort_values(by = ['S=P/C'])
    
    Ai = calc_Ai(df1)
    Ai = connect_Ai(Ai, df_Resi)
    Ai = connect_age(Ai, df_age)
    
    # Ai.shape[0]
    # Ai.loc[Ai['District'] == '永定路街道'].shape[0]
    
    
    kdx_District = sum_pA_District(Ai, str_p)    
    kdx_region = sum_pA_region(Ai, str_p)
    
    kdx_District = connect_p_District(kdx_District, df_Resi)
    kdx_region = connect_p_region(kdx_region, df_Resi)
    
    
    save_add_column()
    
