# -*- coding: utf-8 -*-
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def make_data_for_find_sample_size() :

    musig = [(10,3.64), (10,1.64), (0,1), (14, 1.64), (14,3)]
    n = int(4000000/len(musig))
    df = pd.DataFrame()
    for i, v in enumerate(musig) :
        d = stats.expon.rvs(loc=v[0], scale=v[1], size=n)
        d = pd.DataFrame(d, columns=['TARGET'])
        d['VARIATION'] = 'GROUP' + str(i)
        df = df.append(d)
    return df

def make_data_for_hourly_analysis(norm='all') : # all | some | not

    n = 800000  # variation당 샘플사이즈
    variation_cnt = 5  # variation 수
    sample_size_tot = n * variation_cnt  # 전체 샘플사이즈

    df = pd.DataFrame()
    if norm=='all' :
        # musig = [(10, 1), (10, 2), (10, 3)] # n_variations = 3
        musig = [(10, 1), (11, 1), (12, 2), (10, 3), (15, 3)] # n_variations = 5

        for i, v in enumerate(musig):
            d = stats.norm.rvs(loc=v[0], scale=v[1], size=n)
            d = pd.DataFrame(d, columns=['DECI_IDX_ANAL_VLUE'])
            d['ABTEST_EXP_ID'] = 'V00' + str(i)
            df = df.append(d)

    elif norm=='some' :
        # n_variations = 5
        musig = [(10, 1), (10.01, 1), (12, 2)]
        for i, v in enumerate(musig):
            d = stats.norm.rvs(loc=v[0], scale=v[1], size=n)
            d = pd.DataFrame(d, columns=['DECI_IDX_ANAL_VLUE'])
            d['ABTEST_EXP_ID'] = 'V00' + str(i)
            df = df.append(d)

        d = stats.expon.rvs(loc=10, scale=1, size=n)
        d = pd.DataFrame(d, columns=['DECI_IDX_ANAL_VLUE'])
        d['ABTEST_EXP_ID'] = 'V003'
        df = df.append(d)

        d = stats.expon.rvs(loc=10, scale=1, size=n)
        # d = stats.gamma.rvs(1.99, loc=10, scale=1, size=n)
        d = pd.DataFrame(d, columns=['DECI_IDX_ANAL_VLUE'])
        d['ABTEST_EXP_ID'] = 'V004'
        df = df.append(d)

    else :
        musig = [(10, 2), (10, 2), (10, 2)]  # n_variations = 3
        # musig = [(10, 1), (10, 1), (10, 1), (11, 1), (12, 1)]
        for i, v in enumerate(musig):
            d = stats.expon.rvs(loc=v[0], scale=v[1], size=n)
            d = pd.DataFrame(d, columns=['DECI_IDX_ANAL_VLUE'])
            d['ABTEST_EXP_ID'] = 'V00' + str(i)
            df = df.append(d)

    df.reset_index(drop=True, inplace=True)
    if min(df['DECI_IDX_ANAL_VLUE']) < 0:  # 결정지표 값 양수로 변경
        df['DECI_IDX_ANAL_VLUE'] = df['DECI_IDX_ANAL_VLUE'] + abs(min(df['DECI_IDX_ANAL_VLUE']))
    df['ABTEST_PSG_DAYS'] = 1  # 테스트 경과일수 : 고객별 일평균 결정지표 산출 기준(분모값)
    df['ABTEST_ID'] = 'TEST001'  # AB테스트ID
    df['BASE_DTTM'] = '2024-05-01 12:00:00'  # 결과출력 기준일시
    df['DECI_IDX'] = 'IDX001'  # 결정지표ID

    return df

def visualize(df) :
    plt.figure(figsize=(10, 5))
    vlist = df['ABTEST_EXP_ID'].unique().tolist()
    for v in vlist:
        sns.distplot(
            df[df['ABTEST_EXP_ID'] == v]['DECI_IDX_ANAL_VLUE'] / df[df['ABTEST_EXP_ID'] == v]['ABTEST_PSG_DAYS'],
            label=v)
    plt.legend()
    plt.show()