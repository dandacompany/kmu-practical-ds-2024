# -*- coding: utf-8 -*-
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy import stats, optimize
from itertools import combinations
import numpy as np
import pandas as pd
import math
from abc import abstractmethod, ABCMeta
from .utils import Logger, Timer, Flag, Formatter, Optimizer
import warnings
from .abtest_continous import SolveRequiredSampleSize as SolveRequiredSampleSize_PV
warnings.filterwarnings(action='ignore')

class SolveRequiredSampleSize(SolveRequiredSampleSize_PV) :

    def __init__(
            self, variation_cnt, access_rate, pv_per_uv, available_size, baseline, min_det_eff, target_power,
            confidence, nbr_samples=None, max_cust_nbr=1e+7,
            *args, **kwargs
    ):
        '''
        * Writer : Dante Kwak
        * Description
         - Initiate to Finding Required Sample Size For Chi-Square-Test Class
        * Variables
         - variation_cnt   : 대조/실험군 개수                             (입력 integer)
         - access_rate     : 접속율                                    (입력 float)
         - pv_per_uv       : 인당건 Ratio                              (입력 float)
         - available_size  : 데이터길이, AB테스트 가용인원                  (입력 integer)
         - baseline        : 테스트 설계화면에서 산출되는 기준값              (입력 float)
         - min_det_eff     : 사용자가 테스트 설계화면에서 입력한 최소감지효과(%) (입력 float : 0~1 사이)
         - target_power    : 목표 검정력(%)                             (입력 float : 0~1 사이)
         - confidence      : 신뢰수준(%)                                (입력 float : 0~1 사이)
         - nbr_samples     : 최종샘플사이즈                               (입력 integer)
         - max_cust_nbr    : ABTEST 시스템 최대가용인원                   (입력 integer)
        '''
        # 파라미터 저장
        self.variation_cnt = variation_cnt
        self.access_rate = access_rate
        self.pv_per_uv = pv_per_uv
        self.available_size = available_size  # data길이 = AB테스트 가용인원수
        self.baseline = baseline
        self.min_det_eff = min_det_eff
        self.alternative = 'greater' if min_det_eff >= 0 else 'less'  # '21.8.12 최소감지효과의 음양에 따라 검정방향 변경 (양측검정은 제외)
        self.target_power = target_power
        self.alpha = (1.0 - confidence) / 2
        self.nbr_samples = nbr_samples
        self.max_cust_nbr = max_cust_nbr

        # 초기값 셋업
        self.success = False  # 필요샘플사이즈 산출 성공여부값 초기화
        self.error_code = 'NO_SOL'  # 에러코드값 초기화
        self.required_sample_size = np.NAN  # 필요샘플 사이즈 수렴값 초기화
        self.exp_power = np.NAN  # 예상 검정력 초기화
        self.whether_to_optimize = False  # 최소샘플사이즈 해 산출 수행 여부 flag 초기화

        # 로거 셋업
        self.logger = Logger.spawn('Find required sample size')

    def get_power(self, nbr_samples):
        # https://support.minitab.com/ko-kr/minitab/18/help-and-how-to/statistics/power-and-sample-size/how-to/hypothesis-tests/power-and-sample-size-for-2-proportions/methods-and-formulas/methods-and-formulas/#calculating-sample-size-and-comparison-proportion
        if not self.baseline :
            return 0.0
        p1 = self.baseline
        p2 = min(self.baseline * (1 + self.min_det_eff),1)
        p2 = p2 if p2 < 1 else 1
        pc = (p1 + p2) / 2
        z = stats.norm.ppf(1 - self.alpha / 2)
        co_se = np.sqrt(2 * pc * (1 - pc) / nbr_samples)
        se = np.sqrt(p2 * (1 - p2) / nbr_samples + p1 * (1 - p1) / nbr_samples)
        diff_prop = p2 - p1

        if self.alternative == 'greater':
            greater_z = (- diff_prop + z * co_se) / se
            power = stats.norm.sf(greater_z)
        elif self.alternative == 'less':
            less_z = (- diff_prop - z * co_se) / se
            power = stats.norm.cdf(less_z)
        else :
            greater_z = (- diff_prop + z * co_se) / se
            less_z = (- diff_prop - z * co_se) / se
            power = stats.norm.sf(greater_z) + stats.norm.cdf(less_z)
        return power

    def optimize(self):
        '''
        * Writer : Dante Kwak
        * Description
         - Find Required Sample Size by Newton Method
         - 찾은 해(최소샘플사이즈)는 instance varaible에 저장
        * Returns
         - binary type : True (available), False (not available)
        '''

        self.whether_to_optimize = True

        # Validation 1) data 개수 체크 : data 길이(=AB TEST 가용고객수)가 3 미만이면 에러코드 'DATA_SHRT' 반환
        if self.available_size < 2.5:
            self.error_code = 'DATA_SHRT'
            return self.get_result()

        # Validation 2) 샘플사이즈 2개일때의 검정력이 목표검정력(target_power) 보다 큰 경우, 필요샘플사이즈 값 2로 반환
        two_samples_actual_power = self.get_power(2)
        if two_samples_actual_power - self.target_power > 1e-8:
            self.success = True
            self.error_code = 'NORMAL'
            self.required_sample_size = 2
            self.exp_power = two_samples_actual_power
            return self.get_result()

        # Validation 3) 최소샘플사이즈 해 산출(3가지 메소드)
        for optimizer in Optimizer.get_optimizers():
            opt_result = optimizer(self.get_power, self.target_power, self.available_size, self.variation_cnt)
            self.required_sample_size = opt_result['required_sample_size']
            if opt_result['success']:
                self.success = True
                self.error_code = 'NORMAL'
                self.exp_power = self.get_power(self.required_sample_size)
                return self.get_result()

        # 필요 샘플 사이즈 산출 최종 실패시, AB테스트 최대가용인원 수를 사용하여 예상검정력 산출
        self.required_sample_size = self.max_cust_nbr // self.variation_cnt
        self.exp_power = self.get_power(self.max_cust_nbr)
        return self.get_result()

class FindSampleSize(object) :

    def __init__(self, data, pv_per_uv, *args, **kwargs):
        self.data = data
        self.pv_per_uv = pv_per_uv
        self.available_size = len(self.data)  # data길이 = AB테스트 가용인원수
        self.result = {
            'success': False,
            'error_code': 'CANT_CAL',
            'req_sample_size': None,
            'unit_req_sample_size': None,
            'access_rate': None,
            'exp_power': None,
            'adj_req_sample_size': None,
            'adj_unit_req_sample_size': None,
            'max_cust_nbr': None
        }
        self.solver = SolveRequiredSampleSize

    @Timer(Flag.log)
    def analysis(self, access_rate, variation_id_lst, min_det_eff, target_power, confidence, max_cust_nbr, *args, **kwargs) :

        if access_rate < 0:
            self.result['error_code'] = 'DATA_SHRT'
            return self.result

        if not len(self.data) :
            self.result['error_code'] = 'DATA_SHRT'
            return self.result

        baseline = np.mean(self.data)

        self.anal_params = {
            'access_rate': access_rate,
            'pv_per_uv' : self.pv_per_uv,
            'variation_cnt': len(variation_id_lst),
            'available_size': self.available_size,
            'baseline': baseline,
            'min_det_eff': min_det_eff,
            'target_power': target_power,
            'confidence': confidence,
            'max_cust_nbr': max_cust_nbr,
        }

        nbr_samples = kwargs.get('nbr_samples', None)
        if nbr_samples:
            self.anal_params.update({
                'nbr_samples': nbr_samples
            })

        srss = self.solver(**self.anal_params)

        if nbr_samples:
            # 최종샘플사이즈에 대한 예상검정력 산출
            return {
                'actual_power': srss.get_power(nbr_samples * self.pv_per_uv)
            }
        else:
            # 최소샘플사이즈 산출
            return srss.optimize()

    def get_power(self, min_det_eff, confidence, nbr_samples, *args, **kwargs):
        if not any(self.data) :
            raise Exception('DATA_SHRT')

        baseline = np.mean(self.data)

        srss = self.solver(**{
            'variation_cnt': None,
            'access_rate': None,
            'pv_per_uv' : self.pv_per_uv,
            'available_size': None,
            'baseline': baseline,
            'min_det_eff': min_det_eff,
            'target_power': None,
            'confidence': confidence
        })

        return srss.get_power(nbr_samples * self.pv_per_uv)

class Tester(metaclass=ABCMeta) :

    def __init__(self, dfg, alpha_test, *args, **kwargs):
        self.dfg = dfg
        self.alpha_test = alpha_test
        self.logger = Logger.spawn(self.__str__())

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def __str__(self):
        return "TesterForChiSq"

    @staticmethod
    def calc_proportion_test_ci(data, alpha_test):
        # minitab : https://support.minitab.com/ko-kr/minitab/18/help-and-how-to/statistics/basic-statistics/how-to/2-proportions/methods-and-formulas/methods-and-formulas/
        data = np.array(data)
        n1 = data[0].sum()
        n2 = data[1].sum()
        p1 = data[0,0] / n1
        p2 = data[1,0] / n2

        se = np.sqrt( p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2 )
        diff_prop = p2 - p1
        cl_ll, cl_ul = diff_prop + np.array([-1, 1]) * stats.norm.ppf(1 - alpha_test / 2) * se
        return cl_ll, cl_ul

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def analysis(cls, data, normality, eqvar) :
        pass

    @classmethod
    def calc_power(cls, *args, **kwargs):
        pass

class PriorTester(Tester) :

    def __str__(self):
        return "PriorTesterForChiSq"

    # 2개의 variation 조합에 대한 통계 검정 수행
    def execute(self):

        variations = self.dfg.index.tolist()
        variation_pairs = combinations(variations, 2)
        dfg_tmp = self.dfg.reset_index()[['ABTEST_EXP_ID', 'DECI_VLUE_LIST', 'EXP_AVG_VLUE', 'EXP_STD_VLUE', 'EXP_SAMP_SIZ']].copy()
        dfg_tmp['key'] = 1
        df_base = dfg_tmp.merge(dfg_tmp, on='key').drop('key', 1) # cross merge
        df_base = df_base.set_index(['ABTEST_EXP_ID_x', 'ABTEST_EXP_ID_y']).loc[variation_pairs].reset_index()
        df_base['NORM_TEST_YN'] = None
        df_base['HVARI_TEST_YN'] = None
        df_base['ALPHA_TEST_BASE_VLUE'] = self.alpha_test

        # analysis by variation pairs
        stat_df = pd.DataFrame(df_base.apply(
            lambda df : self.analysis(data=[
                [sum(df['DECI_VLUE_LIST_x']), len(df['DECI_VLUE_LIST_x']) - sum(df['DECI_VLUE_LIST_x'])],
                [sum(df['DECI_VLUE_LIST_y']), len(df['DECI_VLUE_LIST_y']) - sum(df['DECI_VLUE_LIST_y'])],
            ], alpha_test=self.alpha_test), axis=1).tolist(),
            columns=['STAT', 'PV_VLUE', 'CFD_LLMT', 'CFD_HLMT', 'ANAL_ALG_NM'])

        df_base = pd.concat([df_base, stat_df], axis=1)

        # calculate power
        df_base['RSLT_POWER'] = df_base.apply(
            lambda df : self.calc_power(data=[
                [sum(df['DECI_VLUE_LIST_x']), len(df['DECI_VLUE_LIST_x']) - sum(df['DECI_VLUE_LIST_x'])],
                [sum(df['DECI_VLUE_LIST_y']), len(df['DECI_VLUE_LIST_y']) - sum(df['DECI_VLUE_LIST_y'])],
            ], alpha_test=self.alpha_test, nbr_samples=1), axis=1)
        df_base['PV_VLUE'] = df_base['PV_VLUE'].apply(Formatter.pvalue)
        # 분석단계 저장
        df_base['ABTEST_DIVS_NM'] = 'BASE'

        # combinations -> permutations
        df_base_copy = df_base.rename(
            columns={'ABTEST_EXP_ID_x': 'ABTEST_EXP_ID_y', 'ABTEST_EXP_ID_y': 'ABTEST_EXP_ID_x'}).copy()
        result_df = df_base.append(df_base_copy). \
            drop_duplicates(['ABTEST_EXP_ID_x', 'ABTEST_EXP_ID_y']). \
            sort_values(['ABTEST_EXP_ID_x', 'ABTEST_EXP_ID_x']). \
            reset_index(drop=True)

        # rename columns
        result_df = result_df.rename(columns={
            'ABTEST_EXP_ID_x': 'ABTEST_EXP_ID',
            'ABTEST_EXP_ID_y': 'OPRD_EXP_ID',
            'EXP_AVG_VLUE_x': 'EXP_AVG_VLUE',
            'EXP_AVG_VLUE_y': 'OPRD_EXP_AVG_VLUE',
            'EXP_STD_VLUE_x': 'EXP_STD_VLUE',
            'EXP_STD_VLUE_y': 'OPRD_EXP_STD_VLUE',
            'EXP_SAMP_SIZ_x': 'EXP_SAMP_SIZ',
            'EXP_SAMP_SIZ_y': 'OPRD_EXP_SAMP_SIZ',
        })

        # extract necessary columns
        result_df = result_df[[
            'ABTEST_EXP_ID', 'OPRD_EXP_ID', 'NORM_TEST_YN',
            'HVARI_TEST_YN', 'PV_VLUE', 'ALPHA_TEST_BASE_VLUE',
            'RSLT_POWER', 'ABTEST_DIVS_NM', 'ANAL_ALG_NM'
        ]]

        return result_df

    @classmethod
    def analysis(cls, data, alpha_test):

        if np.min(data) > 5 :
            stat, p, df, ex = stats.chi2_contingency(data)
            anal_alg_nm = 'two_proportion_test'

        else :
            stat, p = stats.fisher_exact(data)
            anal_alg_nm = 'fisher_exact_test'

        ci_ll, ci_ul = cls.calc_proportion_test_ci(data=data, alpha_test=alpha_test)

        return stat, p, ci_ll, ci_ul, anal_alg_nm

    @classmethod
    def calc_power(cls, data, alpha_test, nbr_samples):
        '''
        https://support.minitab.com/ko-kr/minitab/18/help-and-how-to/statistics/power-and-sample-size/how-to/hypothesis-tests/power-and-sample-size-for-2-proportions/methods-and-formulas/methods-and-formulas/#calculating-sample-size-and-comparison-proportion
        '''
        data = np.array(data)
        n1 = np.sum(data[0])
        n2 = np.sum(data[1])
        nc = n1 + n2
        p1 = data[0,0] / n1
        p2 = data[1,0] / n2
        pc = (p1 + p2) / 2
        z = stats.norm.ppf(1 - alpha_test / 2)
        co_se = np.sqrt(2 * pc * (1 - pc) / nc)
        se = np.sqrt(p2 * (1 - p2) / n2 + p1 * (1 - p1) / n1)
        diff_prop = p2 - p1
        greater_z = (- diff_prop + z * co_se) / se
        less_z = (- diff_prop - z * co_se) / se
        exp_power = 1 - stats.norm.cdf(greater_z) + stats.norm.cdf(less_z)
        return exp_power

class ChiSqTester(Tester) :

    def __str__(self):
        return "ChiSqTester"

    # 전체 Variations에 대한 통계 검정 수행
    def execute(self):
        data = self.dfg['DECI_VLUE_LIST']
        stat, p, cl_ll, cl_ul, anal_alg_nm = self.analysis(data, alpha_test=self.alpha_test)

        df_anova = pd.DataFrame({
            'ABTEST_EXP_ID': ['DEFAULT'],
            'OPRD_EXP_ID': ['DEFAULT'],
            'NORM_TEST_YN': [None],
            'HVARI_TEST_YN': [None],
            'STAT' : [stat],
            'PV_VLUE': [Formatter.pvalue(p)],
            'ALPHA_TEST_BASE_VLUE': [self.alpha_test],
            'RSLT_POWER': [None],
            'ABTEST_DIVS_NM': ['CHISQ'],
            'ANAL_ALG_NM': [anal_alg_nm],
            'CFD_LLMT': [cl_ll],
            'CFD_HLMT': [cl_ul],
        })

        return df_anova, p, None

    @classmethod
    def analysis(cls, data, alpha_test):
        # chi-square or fisher-exact test
        data = np.array([[sum(d), len(d) - sum(d)] for d in data])
        if data.shape == (2,2) and np.min(data) <= 5 :
            stat, p = stats.fisher_exact(data)
            cl_ll, cl_ul = cls.calc_proportion_test_ci(data, alpha_test=alpha_test)
            anal_alg_nm = 'fisher_exact_test'

        else :
            if data.shape == (2,2) :
                cl_ll, cl_ul = cls.calc_proportion_test_ci(data, alpha_test=alpha_test)
            else :
                cl_ll, cl_ul = np.nan, np.nan

            stat, p, df, ex = stats.chi2_contingency(data)
            anal_alg_nm = 'chi_square_test'

        return stat, p, cl_ll, cl_ul, anal_alg_nm

class PostTester(Tester) :

    def __init__(self, dfg, parametric, *args, **kwargs):
        self.parametric = parametric
        super().__init__(dfg, *args, **kwargs)

    def __str__(self):
        return "PostTesterForChiSq"

    # 두 Variation 조합에 대한 사후분석 수행
    @Timer(Flag.log)
    def execute(self):

        variation_pairs = list(combinations(self.dfg.index, 2))

        # chi-square test 시행  / Bonferroni correction
        alpha_test = self.alpha_test / len(variation_pairs)  # Bonferroni correction : alpha값 조정(variation 조합 수로 나눔)
        dfg_tmp = self.dfg.reset_index()[['ABTEST_EXP_ID', 'DECI_VLUE_LIST']].copy()
        dfg_tmp['key'] = 1
        df_post = dfg_tmp.merge(dfg_tmp, on='key').drop('key', 1) # cross merge
        df_post.set_index(['ABTEST_EXP_ID_x', 'ABTEST_EXP_ID_y'], inplace=True)
        df_post = df_post.loc[variation_pairs]

        stat_df = pd.DataFrame(
            df_post.apply(lambda df: self.analysis([df['DECI_VLUE_LIST_x'], df['DECI_VLUE_LIST_y']], self.alpha_test)
            , axis=1
            ).tolist(),
            columns=['STAT', 'PV_VLUE', 'CFD_LLMT', 'CFD_HLMT', 'ANAL_ALG_NM'],
            index=df_post.index
        )
        df_post = pd.concat([df_post, stat_df], axis=1)
        df_post.PV_VLUE = multipletests(df_post.PV_VLUE, method='bonf', alpha=alpha_test)[1]
        df_post['PV_VLUE'] = df_post['PV_VLUE'].apply(Formatter.pvalue)

        df_post['ALPHA_TEST_BASE_VLUE'] = alpha_test
        df_post = df_post.reset_index().drop(columns=['DECI_VLUE_LIST_x', 'DECI_VLUE_LIST_y'])

        # 기준 variation과 비교 variation만 변경하고 동일 결과 copy
        df_post_copy = df_post.rename(
            columns={'ABTEST_EXP_ID_x': 'ABTEST_EXP_ID_y', 'ABTEST_EXP_ID_y': 'ABTEST_EXP_ID_x'}).copy()

        df_post = df_post.append(df_post_copy). \
            drop_duplicates(['ABTEST_EXP_ID_x', 'ABTEST_EXP_ID_y']). \
            sort_values(['ABTEST_EXP_ID_x', 'ABTEST_EXP_ID_x']). \
            reset_index(drop=True)

        df_post = df_post.rename(columns={
            'ABTEST_EXP_ID_x': 'ABTEST_EXP_ID',
            'ABTEST_EXP_ID_y': 'OPRD_EXP_ID'
        })

        df_post['ABTEST_DIVS_NM'] = 'POST'

        return df_post

    @classmethod
    def analysis(cls, data, alpha_test):
        # chi-square or fisher-exact test
        data = np.array([[sum(d), len(d) - sum(d)] for d in data])

        if data.shape == (2, 2) and np.min(data) <= 5:
            stat, p = stats.fisher_exact(data)
            anal_alg_nm = 'fisher_exact_test'

        else:
            stat, p, df, ex = stats.chi2_contingency(data)
            anal_alg_nm = 'chi_square_test'

        ci_ll, ci_ul = cls.calc_proportion_test_ci(data, alpha_test)
        return stat, p, ci_ll, ci_ul, anal_alg_nm

class BayesianTester(Tester) :

    def __init__(self, dfg, *args, **kwargs):
        self.dfg = dfg

    def execute(self):

        df_bayes = self.analysis(self.dfg)
        df_bayes['ABTEST_DIVS_NM'] = 'BAYES'
        return df_bayes

    def analysis(self, dfg, tr_col='EXP_SAMP_SIZ', cv_col='EXP_ACUM_VLUE', precision=3):
        anal_alg_nm = ['bayesian (binomial dist)']
        result = [None] * dfg.shape[0]
        decay = 1 - pow(10, -precision + 1)  # 전환율 100%일 경우 계산오류 방어용 decay rate
        for idx, orig_grp in enumerate(dfg.index):
            comp_dfg = dfg.drop(index=orig_grp)

            def f(z):
                ca = dfg.loc[orig_grp, cv_col]
                ta = dfg.loc[orig_grp, tr_col]
                ca *= decay if ca == ta else 1
                r = stats.beta.pdf(z, ca + 1, ta - ca + 1)
                for comp_grp in comp_dfg.index:
                    cb = comp_dfg.loc[comp_grp, cv_col]
                    tb = comp_dfg.loc[comp_grp, tr_col]
                    cb *= decay if cb == tb else 1
                    r *= stats.beta.cdf(z, cb + 1, tb - cb + 1)
                return r

            prob = np.mean([f(x) for x in np.linspace(0, 1, pow(10, precision))])  # 적분
            result[idx] = Formatter.percentage(prob)

        return pd.DataFrame({"PV_VLUE": result, "ANAL_ALG_NM": anal_alg_nm}, index=dfg.index).reset_index()

