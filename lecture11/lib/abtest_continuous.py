# -*- coding: utf-8 -*-
from scipy import stats, optimize, integrate
from statsmodels.sandbox.stats.multicomp import MultiComparison
from itertools import combinations
import numpy as np
import pandas as pd
import math
from abc import abstractmethod, ABCMeta
from .utils import Logger, Timer, Flag, Formatter, Optimizer
import warnings

warnings.filterwarnings(action='ignore')


class SolveRequiredSampleSize(object):

    def __init__(
            self, variation_cnt, access_rate, var, available_size, baseline, min_det_eff, target_power,
            confidence, nbr_samples=None, max_cust_nbr=1e+7,
            *args, **kwargs
    ):

        '''
        * Writer : Dante. Kwak
        * Description
         - Initiate to Finding Required Sample Test Size Class
        * Variables
         - variation_cnt   : 그룹수                                    (입력 integer)
         - access_rate     : 접속률                                    (입력 float)
         - var             : 모분산                                    (입력 float)
         - available_size  : 데이터길이, AB테스트 가용인원                  (입력 integer)
         - baseline        : 테스트 설계화면에서 산출되는 기준값              (입력 float)
         - min_det_eff     : 사용자가 테스트 설계화면에서 입력한 최소감지효과(%) (입력 float : 0~1 사이) -> (입력 float : -1~1 사이로 변경, 단측처리를 위해)
         - target_power    : 목표 검정력(%)                             (입력 float : 0~1 사이)
         - confidence      : 신뢰수준(%)                                (입력 float : 0~1 사이)
         - nbr_samples     : 최종샘플사이즈                               (입력 integer)
         - max_cust_nbr    : ABTEST 시스템 최대가용인원                   (입력 integer)
        '''
        # 파라미터 저장
        self.variation_cnt = variation_cnt
        self.access_rate = access_rate
        self.var = var  # 데이터(data) 모분산 산출
        self.available_size = available_size  # data길이 = AB테스트 가용인원수
        self.baseline = baseline
        self.min_det_eff = min_det_eff
        self.alternative = 'greater' if min_det_eff >= 0 else 'less' # 최소감지효과의 음양에 따라 검정방향 변경 (양측검정은 제외)
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
        '''
        * Writer : Dante. Kwak
        * Description
         - Calculate Test Power
        * Variables
         - is_normal       : 정규성여부                                   (입력 binary)
         - baseline        : 테스트 설계화면에서 산출되는 기준값                (입력 integer)
         - min_det_eff     : 사용자가 테스트 설계화면에서 입력한 최소감지효과      (입력 float : 0~1)
         - alternative     : 검정방향(오른쪽, 왼쪽)                           (입력 str : ['greater'|'less'])
         - alpha_norm      : 정규성 검정시 사용되는 alpha                    (입력 float : 0~1)
         - var             : 모분산                                      (입력 float)
         - nbr_samples     : 샘플사이즈                                   (입력 integer)
        * Returns
         - 0~1 float type number or np.NAN
        '''

        """ 비모수검정은 샘플사이즈산출에서 제외
        if not self.is_normal:
            '''
            is_normal = True  "Calculate power by STUDENT'T TEST"
            is_normal = False "Calculate power by MANN'S WHITNEY U TEST"
            '''
            # mann-whitney 검정 기법 기준으로 샘플 사이즈 산출시, 'nbr_samples'를 조정
            # ref : https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Mann-Whitney_U_or_Wilcoxon_Rank-Sum_Tests_for_Non-Inferiority.pdf
            nbr_samples = nbr_samples / (np.pi / 3)  # adjustment
        """
        # 자유도(degree of freedom) 산출
        dof = 2 * nbr_samples - 2
        # t.isf(ALPHA, deg_freedom) = 임계값(deg_freedom가 주어질때 t분포에서 ALPHA가 오른쪽 기각역 일때 임계값)
        crit = stats.t.isf(self.alpha, dof)
        # 비중심 모수에 사용할 모수값(non-centrality parameter) 산출
        parameter = self.baseline * self.min_det_eff * np.sqrt(nbr_samples / 2) / np.sqrt(self.var)
        # 샘플 사이즈가 nbr_samples 일때의 실제 검정력 계산
        if self.alternative == 'greater':
            power = stats.nct.sf(crit, dof, parameter)
        elif self.alternative == 'less':
            power = stats.nct.cdf(-crit, dof, parameter)
        else :
            power = stats.nct.sf(crit, dof, parameter) + stats.nct.cdf(-crit, dof, parameter)
        return power


    @Timer(Flag.log)
    def optimize(self):
        '''
        * Writer : Dante. Kwak
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

        # Validation 2) 분산(var)이 'get_power' 함수 내 계산식에서 분모에 들어가므로, var이 0이면 에러코드 'ZERO_VAR' 반환
        if self.var < 1e-8:
            self.error_code = 'ZERO_VAR'
            return self.get_result()

        # Validation 3) 샘플사이즈 2개일때의 검정력이 목표검정력(target_power) 보다 큰 경우, 필요샘플사이즈 값 2로 반환
        two_samples_actual_power = self.get_power(2)
        if two_samples_actual_power - self.target_power > 1e-8:
            self.success = True
            self.error_code = 'NORMAL'
            self.required_sample_size = 2
            self.exp_power = two_samples_actual_power
            return self.get_result()

        # Validation 4) 최소샘플사이즈 해 산출(3가지 메소드)
        for optimizer in Optimizer.get_optimizers():
            opt_result = optimizer(self.get_power, self.target_power, self.available_size, self.variation_cnt )
            self.required_sample_size = opt_result['required_sample_size']
            if opt_result['success']:
                self.success = True
                self.error_code = 'NORMAL'
                self.exp_power = self.get_power(self.required_sample_size)
                return self.get_result()

        # 필요 샘플 사이즈 산출 최종 실패시, AB테스트 최대가용인원 수를 사용하여 예상검정력 산출
        self.required_sample_size = self.max_cust_nbr
        self.exp_power = self.get_power(self.max_cust_nbr)
        return self.get_result()

    def get_result(self):
        '''
        * Writer : Dante. Kwak

        * Description
         - Re 최소샘플사이즈 수행결과 리턴 함수

        * Returns
         - 4 factor tuple type
         - success                  : 성공여부                                (출력 integer)
         - error_code               : 에러코드                                (출력 string : NO_SOL(해없음), NORMAL(성공), DATA_SHRT(가용고객수2미만), ZERO_VAR(제로분산))
         - req_sample_size          : 필요샘플사이즈                           (출력 integer)
         - unit_req_sample_size     : Variation 당 필요샘플사이즈               (출력 integer)
         - adj_req_sample_size      : 필요샘플사이즈(트렌드반영)                  (출력 integer)
         - adj_unit_req_sample_size : Variation 당 필요샘플사이즈(트렌드반영)      (출력 integer)
         - exp_power                : 목표(예상)검정력                          (출력 float)
        '''

        result = {
            'success': self.success,
            'error_code': self.error_code,
            'req_sample_size': self.required_sample_size if not np.isnan(self.required_sample_size) else None,
            'unit_req_sample_size': (self.required_sample_size // self.variation_cnt) if not np.isnan(self.required_sample_size) else None,
            'access_rate': self.access_rate,
            'exp_power': round(self.exp_power, 3) if not np.isnan(self.exp_power) else None
        }
        adj_unit_req_sample_size = int(np.ceil(result['unit_req_sample_size'] / self.access_rate)) if result['unit_req_sample_size'] else None
        result.update({
            'adj_req_sample_size': (adj_unit_req_sample_size * self.variation_cnt) if result['req_sample_size'] else None,
            'adj_unit_req_sample_size': adj_unit_req_sample_size,
            'max_cust_nbr': self.max_cust_nbr
        })
        return result

class FindSampleSize(object):

    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.available_size = len(self.data)  # data길이 = AB테스트 가용인원수
        self.solver = SolveRequiredSampleSize
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

    @Timer(Flag.log)
    def analysis(self, access_rate, variation_id_lst, min_det_eff, target_power, confidence,
                 max_cust_nbr, *args, **kwargs):

        if len(self.data) < 3:
            return self.result

        if access_rate < 0:
            self.result['error_code'] = 'DATA_SHRT'
            return self.result

        var = np.var(self.data),  # 모분산
        var = var[0] if isinstance(var, tuple) else var
        baseline = np.mean(self.data)  # 기준값 여부
        self.anal_params = {
            'access_rate': access_rate,
            'variation_cnt': len(variation_id_lst),
            'available_size': self.available_size,
            'baseline': baseline,
            'min_det_eff': min_det_eff,
            'target_power': target_power,
            'confidence': confidence,
            'max_cust_nbr': max_cust_nbr,
            'var': var
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
                'actual_power': srss.get_power(nbr_samples)
            }
        else:
            # 최소샘플사이즈 산출
            return srss.optimize()

    def get_power(self, min_det_eff, confidence, nbr_samples, *args, **kwargs):

        if not any(self.data) :
            raise Exception('DATA_SHRT')
        var = np.var(self.data),  # 모분산
        var = var[0] if isinstance(var, tuple) else var

        # Validation ) 분산(var)이 'get_power' 함수 내 계산식에서 분모에 들어가므로, var이 0이면 에러코드 'ZERO_VAR' 반환
        if var < 1e-8:
            raise Exception('ZERO_VAR')
        baseline = np.mean(self.data)  # 기준값

        srss = self.solver(**{
            'variation_cnt': None,
            'access_rate': None,
            'var': var,
            'available_size': None,
            'baseline': baseline,
            'min_det_eff': min_det_eff,
            'target_power': None,
            'confidence': confidence
        })

        return srss.get_power(nbr_samples)

class Tester(metaclass=ABCMeta):

    def __init__(self, dfg, alternative='greater', n_normality=None, alpha_test=None, alpha_norm=None, alpha_eqvar=None, *args, **kwargs):
        self.dfg = dfg
        self.alternative = alternative
        self.alpha_test = alpha_test
        self.alpha_norm = alpha_norm
        self.alpha_eqvar = alpha_eqvar
        self.n_normality = n_normality
        self.logger = Logger.spawn(self.__str__())

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def __str__(self):
        return "Tester"

    # 정규성 검정
    @staticmethod
    def normality(data, n_normality, alpha_norm, mean, std):
        '''
        * Writer : Dante. Kwak
        * Description
         - Test Normality
         - variation별 실제샘플사이즈 기준으로 shapiro or ks test 결정
         - python warning : N=5000 (N<=5000일 때, shapiro 가능; warning 표시하지않음)
         - kstest에서는 args=(mean, std) input para 입력 필요
        * Variables
         - data            : AB테스트 가용인원에 대한 기준값 산출 기간의 일자별 집계합의 일평균 결정지표값   (입력 list, tuple, series, array 등의 1차원 iterative data set)
         - n_normality     : 정규성 검정 알고리즘에서 shapiro or ks 선택 기준이 되는 파라미터           (입력 integer)
         - alpha           : 정규성 검정시 사용되는 alpha                                         (입력 float : 0~1)
        * Returns
         - binary type : True (normal), False (not normal)
        '''
        if len(data) <= n_normality:  # shapiro test
            stat, p = stats.shapiro(data)
        else:  # ks test
            stat, p = stats.kstest(data, 'norm', args=(mean, std))

        return p >= alpha_norm

    # 등분산성 검정
    @staticmethod
    def equal_variance(data, alpha_eqvar):
        '''
        * Writer : Dante. Kwak
        * Description
         - Test Equal Variance
        * Variables
         - data     : 연속형 실수의 1차원 배열 데이터    (입력 list, tuple, series, array 등의 1차원 iterative data set)
         - alpha    : 유의수준                      (입력 float : 0~1)
        * Returns
         - binary type : True (equal variance), False (unequal variance)
         - levene test, brown forsythe test 시행 후 최소 한 개 이상 만족시 True
        '''
        stat1, p1 = stats.levene(*data, center='mean')
        stat2, p2 = stats.levene(*data, center='median')

        return p1 >= alpha_eqvar or p2 >= alpha_eqvar

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def analysis(cls, data, normality, eqvar):
        pass

    @classmethod
    def calc_power(cls, *args, **kwargs):
        pass

class PriorTester(Tester):

    def __str__(self):
        return "PriorTester"

    # 2개의 variation 조합에 대한 통계 검정 수행
    def execute(self):
        variations = self.dfg.index.tolist()
        variation_pairs = combinations(variations, 2)
        dfg_tmp = self.dfg.reset_index()[
            ['ABTEST_EXP_ID', 'DECI_VLUE_LIST', 'EXP_AVG_VLUE', 'EXP_STD_VLUE', 'EXP_SAMP_SIZ', 'NORM']].copy()
        dfg_tmp['key'] = 1
        df_base = dfg_tmp.merge(dfg_tmp, on='key').drop('key', 1)  # cross merge
        df_base = df_base.set_index(['ABTEST_EXP_ID_x', 'ABTEST_EXP_ID_y']).loc[variation_pairs].reset_index()
        df_base['NORM_TEST_YN'] = df_base['NORM_x'] & df_base['NORM_y']  # 2개의 variation 결정지표 데이터가 모두 정규성을 만족하는지 판단
        df_base.drop(columns=['NORM_x', 'NORM_y'], inplace=True)
        df_base['HVARI_TEST_YN'] = df_base.apply(  # 등분산성 검정
            lambda df: self.equal_variance(df[['DECI_VLUE_LIST_x', 'DECI_VLUE_LIST_y']], self.alpha_eqvar), axis=1)
        df_base['ALPHA_TEST_BASE_VLUE'] = self.alpha_test
        # analysis by variation pairs
        stat_df = pd.DataFrame(df_base.apply(
            lambda df: self.analysis(
                df[['DECI_VLUE_LIST_y', 'DECI_VLUE_LIST_x']],
                df['HVARI_TEST_YN'],
                self.alpha_test,
                self.alternative
            ), axis=1).tolist(),
            columns=['STAT', 'PV_VLUE', 'CFD_LLMT', 'CFD_HLMT', 'ANAL_ALG_NM']
        )
        df_base = pd.concat([df_base, stat_df], axis=1)
        df_base['PV_VLUE'] = df_base['PV_VLUE'].apply(Formatter.pvalue)
        # calculate power
        df_base['RSLT_POWER'] = df_base.apply(lambda df: self.calc_power(
            mu1=df['EXP_AVG_VLUE_x'], mu2=df['EXP_AVG_VLUE_y'],
            s1=df['EXP_STD_VLUE_x'], s2=df['EXP_STD_VLUE_y'],
            s=np.std(df['DECI_VLUE_LIST_x'] + df['DECI_VLUE_LIST_y']),
            n1=np.size(df['DECI_VLUE_LIST_x']), n2=np.size(df['DECI_VLUE_LIST_y']),
            eqvar=df['HVARI_TEST_YN'], alternative=self.alternative, alpha_test=df['ALPHA_TEST_BASE_VLUE']
        ), axis=1)
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
            'RSLT_POWER', 'ABTEST_DIVS_NM', 'ANAL_ALG_NM', 'CFD_LLMT', 'CFD_HLMT'
        ]]
        return result_df

    @classmethod
    def analysis(cls, data, eqvar, alpha_test, alternative):

        (n1, mu1, v1), (n2, mu2, v2) = [(len(d), np.mean(d), np.var(d)) for d in data]
        if eqvar:  # 등분산성 만족 (student's t-test)
            df = n1 + n2 - 2
            cv = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
            denom = np.sqrt(cv * (1.0 / n1 + 1.0 / n2))
            anal_alg_nm = 'student_t_test'
        else:  # 등분산성 불만족 (welch's t-test)
            vn1 = v1 / n1
            vn2 = v2 / n2
            with np.errstate(divide='ignore', invalid='ignore'):
                df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))
            df = np.where(np.isnan(df), 1, df)
            denom = np.sqrt(vn1 + vn2)
            anal_alg_nm = 'welch_t_test'

        effect_size = mu2 - mu1
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = np.divide(effect_size, denom)

        if alternative == 'greater' :
            p = stats.t.sf(stat, df)
        elif alternative == 'less' :
            p = stats.t.cdf(stat, df)
        else :
            p = stats.t.sf(stat, df) + stats.t.cdf(stat, df)

        ci_ll = effect_size - stats.t.ppf(1 - alpha_test / 2, df) * denom
        ci_ul = effect_size + stats.t.ppf(1 - alpha_test / 2, df) * denom

        return stat, p, ci_ll, ci_ul, anal_alg_nm


    @classmethod
    def calc_power(cls, mu1, mu2, s1, s2, s, n1, n2, eqvar, alternative, alpha_test):

        if not s:  # 분산이 0일 경우, 검정력 계산 불가능 (에러코드 필요)
            return np.nan

        if not n1 or not n2 : # 데이터 사이즈가 0일 때, 검정력 계산 불가능
            return np.nan

        effect_size = mu2 - mu1  # 평균 차이

        # student's t test
        if eqvar:
            dof = n1 + n2 - 2  # 자유도(degree of freedom)
            nc = effect_size / s / np.sqrt(1 / n1 + 1 / n2)  # 비중심모수(non-centrality parameter)
            # nct.cdf(crit, dof, nc) : 자유도 dof, 비중심모수 nc인 non-central t분포에서 x=critical value일 때 cumulative density function 값

            if alternative == 'greater':
                crit = stats.t.isf(alpha_test, dof) # critical value 란 ? : 주어진 유의수준(alpha_test)에서 귀무가설을 기각할 수 있는 경계값
                power = stats.nct.sf(crit, dof, nc) # 
            elif alternative == 'less':
                crit = stats.t.isf(alpha_test, dof)
                power = stats.nct.cdf(-crit, dof, nc)
            else:
                crit = stats.t.isf(alpha_test / 2, dof)
                power = stats.nct.sf(crit, dof, nc) + stats.nct.cdf(-crit, dof, nc)

        # welch's t test
        else :
            if not s1 and not s2:  # 두 variation 분산이 모두 0인 경우, 검정력 계산 불가
                return np.nan

            dof_num = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
            dof_denom = s1 ** 4 / n1 ** 2 / (n1 - 1) + s2 ** 4 / n2 ** 2 / (n2 - 1)
            dof = dof_num / dof_denom
            nc = effect_size / np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)

            if alternative == 'greater':
                crit = stats.t.isf(alpha_test, dof)
                power = stats.nct.sf(crit, dof, nc)
            elif alternative == 'less':
                crit = stats.t.isf(alpha_test, dof)
                power = stats.nct.cdf(-crit, dof, nc)
            else:
                crit = stats.t.isf(alpha_test/2, dof)
                power = stats.nct.sf(crit, dof, nc) + stats.nct.cdf(-crit, dof, nc)

        return power
        # # mann_whitney u test
        # else:
        #     n1 = n1 / (np.pi / 3)
        #     n2 = n2 / (np.pi / 3)
        #     dof = n1 + n2 - 2
        #     nc = d / s / np.sqrt(1 / n1 + 1 / n2)
        #     greater = stats.nct.cdf(stats.t.isf(alpha_test / 2, dof), dof, nc)
        #     less = stats.nct.cdf(-stats.t.isf(alpha_test / 2, dof), dof, nc)
        #     power = 1 - greater + less


class ANOVATester(Tester):

    def __init__(self, dfg, typ, *args, **kwargs):
        self.typ = typ
        super().__init__(dfg, *args, **kwargs)

    def __str__(self):
        return "ANOVATester"

    # 전체 Variations에 대한 통계 검정 수행
    def execute(self):
        data = self.dfg['DECI_VLUE_LIST']
        normality = all(self.dfg['NORM'])  # 정규성 검정 True & 등분산성 검정 True인 경우에만 변경됨
        eqvar = self.equal_variance(data, self.alpha_eqvar) if normality else False  # 정규성 검정 False인 경우 등분산성 검정 진행하지 않음(eqvar=false)
        parametric = normality and eqvar  # 정규성 검정 True & 등분산성 검정 True인 경우에만 True로 변경됨, parametric 기준으로 사후분석 수행
        stat, p, anal_alg_nm, anal_step = self.analysis(data, typ=self.typ)  # 정규성 여부, 등분산성 여부에 따라 통계 검정 수행
        power = self.calc_power(self.dfg, self.alpha_test, stat, self.typ)  # 테스트 알고리즘별 검정력 계산

        df_anova = pd.DataFrame({
            'ABTEST_EXP_ID': ['DEFAULT'],
            'OPRD_EXP_ID': ['DEFAULT'],
            'NORM_TEST_YN': [normality],
            'HVARI_TEST_YN': [eqvar],
            'PV_VLUE': [Formatter.pvalue(p)],
            'ALPHA_TEST_BASE_VLUE': [self.alpha_test],
            'RSLT_POWER': [power],
            'ABTEST_DIVS_NM': [anal_step],
            'ANAL_ALG_NM': [anal_alg_nm],
            'CFD_LLMT': [None],
            'CFD_HLMT': [None]
        })

        return df_anova, p, parametric

    @classmethod
    def analysis(cls, data, typ='anova'):
        # anova or kruskal-wallis test
        try:
            if typ == 'anova' :
                stat, p = stats.f_oneway(*data)
                anal_alg_nm = 'anova'
                anal_step = 'ANOVA'
            else :
                stat, p = stats.kruskal(*data)
                anal_alg_nm = 'kruskal_wallis_test'
                anal_step = 'KRUSKAL'
        except ValueError:
            stat, p, anal_alg_nm, anal_step = None, None, None, None

        return stat, p, anal_alg_nm, anal_step

    @classmethod
    def calc_power(cls, dfg, alpha_test, stat=0.0, typ='anova'):
        # oneway-anova
        if typ == 'anova':
            s = np.std(np.sum(dfg['DECI_VLUE_LIST']))
            if not s:
                return np.nan
            df_num = dfg.shape[0] - 1
            df_denom = np.sum(dfg['EXP_SAMP_SIZ']) - dfg.shape[0]
            crit = stats.f.isf(alpha_test, df_num, df_denom)
            mu = np.sum(dfg['EXP_SAMP_SIZ'] * dfg['EXP_AVG_VLUE']) / np.sum(dfg['EXP_SAMP_SIZ'])
            nc = np.sum(dfg['EXP_SAMP_SIZ'] * (dfg['EXP_AVG_VLUE'] - mu) ** 2) / s ** 2
            power = stats.ncf.sf(crit, df_num, df_denom, nc)

        # kruskal-wallis
        else:
            dof = dfg.shape[0] - 1
            crit = stats.chi2.isf(alpha_test, dof)
            nc = stat  # nc=Kruskal-Wallis H statistic(test statistic)과 동일하게 사용
            try:
                power = stats.ncx2.sf(crit, dof, nc)
            except TypeError:
                power = 0
        return power

class PostTester(Tester):

    def __init__(self, dfg, parametric, *args, **kwargs):
        self.parametric = parametric
        super().__init__(dfg=dfg, *args, **kwargs)

    def __str__(self):
        return "PostTester"

    # 두 Variation 조합에 대한 사후분석 수행
    @Timer(Flag.log)
    def execute(self):
        if self.parametric:  # Tukey test 진행 (anova 결과가 유의미함)
            data = self.dfg['DECI_VLUE_LIST'].sum()
            abtest_exp_ids = self.dfg['DECI_VLUE_LIST'].apply(len).reset_index().apply(
                lambda df: [df['ABTEST_EXP_ID']] * df['DECI_VLUE_LIST'], axis=1).sum()
            res = MultiComparison(data, abtest_exp_ids).tukeyhsd(alpha=self.alpha_test)
            df_post = pd.DataFrame(res.summary()[1:], columns=[str(x) for x in res.summary()[0]]).astype(str)  # 데이터 타입 str 일괄 변경
            df_post.rename(columns={
                'group1': 'ABTEST_EXP_ID',
                'group2': 'OPRD_EXP_ID',
                'p-adj': 'PV_VLUE',
                'lower': 'CFD_LLMT',
                'upper': 'CFD_HLMT'
            }, inplace=True)
            df_post['PV_VLUE'] = df_post['PV_VLUE'].apply(Formatter.pvalue)
            df_post['STAT'] = None
            df_post['ALPHA_TEST_BASE_VLUE'] = self.alpha_test
            df_post = df_post[
                ['STAT', 'ABTEST_EXP_ID', 'OPRD_EXP_ID', 'PV_VLUE', 'ALPHA_TEST_BASE_VLUE', 'CFD_LLMT', 'CFD_HLMT']]

            # 기준 variation과 비교 variation만 변경하고 동일 결과 copy
            df_post_copy = df_post.copy()
            df_post_copy.rename(columns={
                'OPRD_EXP_ID': 'ABTEST_EXP_ID',
                'ABTEST_EXP_ID': 'OPRD_EXP_ID'
            }, inplace=True)

            df_post = df_post.append(df_post_copy). \
                sort_values(['ABTEST_EXP_ID', 'OPRD_EXP_ID']). \
                reset_index(drop=True)

            df_post['ABTEST_DIVS_NM'] = 'POST'
            df_post['ANAL_ALG_NM'] = "tukey_test"

        else:  # Bonferroni correction(Mann Whitney test)진행 (kruskal-wallis test 결과가 유의미함)

            variations = self.dfg.index.tolist()
            variation_pairs = list(combinations(variations, 2))
            alpha_test = self.alpha_test / len(variation_pairs)  # Bonferroni correction
            dfg_tmp = self.dfg.reset_index()[['ABTEST_EXP_ID', 'DECI_VLUE_LIST']].copy()
            dfg_tmp['key'] = 1
            df_post = dfg_tmp.merge(dfg_tmp, on='key').drop('key', 1)  # cross merge
            df_post = df_post.set_index(['ABTEST_EXP_ID_x', 'ABTEST_EXP_ID_y']).loc[variation_pairs].reset_index()

            # mann_whitney_test 시행
            stat_df = pd.DataFrame(
                df_post.apply(lambda df:
                              self.analysis(
                                  [df['DECI_VLUE_LIST_y'], df['DECI_VLUE_LIST_x']],
                                  alpha_test,
                                  self.alternative
                              ), axis=1
                              ).values.tolist(),
                columns=['STAT', 'PV_VLUE', 'CFD_LLMT', 'CFD_HLMT', 'ANAL_ALG_NM'],
                index=df_post.index
            )
            df_post = pd.concat([df_post, stat_df], axis=1)
            df_post['PV_VLUE'] = df_post['PV_VLUE'].apply(Formatter.pvalue)
            df_post['ALPHA_TEST_BASE_VLUE'] = alpha_test
            df_post = df_post.drop(columns=['DECI_VLUE_LIST_x', 'DECI_VLUE_LIST_y'])

            # 기준 variation과 비교 variation만 변경하고 동일 결과 copy
            df_post_copy = df_post.rename(
                columns={'ABTEST_EXP_ID_x': 'ABTEST_EXP_ID_y', 'ABTEST_EXP_ID_y': 'ABTEST_EXP_ID_x'}).copy()

            df_post = df_post.append(df_post_copy)
            df_post = df_post.sort_values(['ABTEST_EXP_ID_x', 'ABTEST_EXP_ID_x'])
            df_post = df_post.rename(columns={
                'ABTEST_EXP_ID_x': 'ABTEST_EXP_ID',
                'ABTEST_EXP_ID_y': 'OPRD_EXP_ID'
            })
            df_post['ABTEST_DIVS_NM'] = 'NONPARAM_POST'

        return df_post

    @classmethod
    def analysis(cls, data, alpha_test, alternative):
        try:
            stat, p = MannWhitney(data[1], data[0], tail=alternative, sig=alpha_test).test()
            ci_ll, ci_ul = np.nan, np.nan # 계산문제로 제공하지 않기로 함.
            anal_alg_nm = 'mann_whitney_u_test_with_FWER'
            return stat, p, ci_ll, ci_ul, anal_alg_nm
        except Exception as e:
            # 두 variation의 결정지표 값이 하나의 값으로 일치할 때 Mann Whitney Test 수행 불가
            ## ex. V001 결정지표 데이터가 [1, 1, 1], V002 결정지표 데이터가 [1, 1, 1]
            anal_alg_nm = str(e)
            return np.nan, np.nan, np.nan, np.nan, anal_alg_nm

class BayesianTester(Tester) :

    def __init__(self, dfg, *args, **kwargs):
        self.dfg = dfg

    def execute(self):

        df_bayes = self.analysis(self.dfg)
        df_bayes['ABTEST_DIVS_NM'] = 'BAYES'
        return df_bayes

    def analysis(self, dfg, precision=3):
        result = [None] * dfg.shape[0]

        anal_alg_nm = 'bayesian (student_t_dist)'

        for idx, orig_grp in enumerate(dfg.index):
            comp_dfgs = dfg.drop(index=orig_grp)
            mu = np.mean(dfg.loc[orig_grp]['DECI_VLUE_LIST'])
            sd = np.std(dfg.loc[orig_grp]['DECI_VLUE_LIST'])
            dof = len(dfg.loc[orig_grp]['DECI_VLUE_LIST']) - 1

            def f(z, mu, sd, dof):
                r = stats.t.pdf(z, loc=mu, scale=sd, df=dof)
                for comp_grp in comp_dfgs.index:
                    mu2 = np.mean(self.dfg.loc[comp_grp]['DECI_VLUE_LIST'])
                    sd2 = np.std(self.dfg.loc[comp_grp]['DECI_VLUE_LIST'])
                    dof2 = len(self.dfg.loc[comp_grp]['DECI_VLUE_LIST']) - 1
                    r *= stats.t.cdf(z, loc=mu2, scale=sd2, df=dof2)
                return r

            prob = integrate.quad(lambda x: f(x, mu, sd, dof), -np.inf, np.inf)[0]  # 적분
            result[idx] = Formatter.percentage(prob)

        return pd.DataFrame({"PV_VLUE": result, "ANAL_ALG_NM": anal_alg_nm}, index=dfg.index).reset_index()

class MannWhitney():
    '''
    implemented by Dante Kwak.
    '''

    def __init__(self, data1, data2=None, tail='two-sided', sig=0.05):

        self.data1 = data1
        self.data2 = data1 if data2 is None else data1
        self.tail = tail
        self.sig = sig

        self.n1 = len(data1)
        self.n2 = len(data2)

        self.crit_05 = pd.DataFrame(
            {'2': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                   3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0],
             '3': [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0,
                   8.0, 9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 17.0,
                   17.0, 18.0, 18.0],
             '4': [-1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 11.0, 12.0, 13.0,
                   13.0, 15.0, 16.0, 17.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.0, 25.0, 26.0, 27.0, 28.0,
                   29.0, 30.0, 31.0, 31.0],
             '5': [-1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0, 14.0, 15.0, 17.0, 18.0, 19.0,
                   20.0, 22.0, 23.0, 24.0, 25.0, 27.0, 28.0, 29.0, 30.0, 32.0, 33.0, 34.0, 35.0, 37.0, 38.0, 39.0, 40.0,
                   41.0, 43.0, 44.0, 45.0],
             '6': [-1.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0, 11.0, 13.0, 14.0, 16.0, 17.0, 19.0, 21.0, 22.0, 24.0, 25.0,
                   27.0, 29.0, 30.0, 32.0, 33.0, 35.0, 37.0, 38.0, 40.0, 42.0, 43.0, 45.0, 46.0, 48.0, 50.0, 51.0, 53.0,
                   55.0, 56.0, 58.0, 59.0],
             '7': [-1.0, 1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
                   32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0,
                   66.0, 68.0, 70.0, 72.0, 74.0],
             '8': [0, 2, 4, 6, 7, 10, 13, 15, 17, 19, 22, 24, 26, 29, 31, 34, 36, 38, 41, 43, 45, 48, 50, 53, 55, 57,
                   60, 62, 65, 67, 69, 72, 74, 77, 79, 81, 84, 86, 89],
             '9': [0, 2, 4, 7, 10, 12, 15, 17, 20, 23, 26, 28, 31, 34, 37, 39, 42, 45, 48, 50, 53, 56, 59, 62, 64, 67,
                   70, 73, 76, 78, 81, 84, 87, 89, 92, 95, 98, 101, 103],
             '10': [0, 3, 5, 8, 11, 14, 17, 20, 23, 26, 29, 33, 36, 39, 42, 45, 48, 52, 55, 58, 61, 64, 67, 71, 74, 77,
                    80, 83, 87, 90, 93, 96, 99, 103, 106, 109, 112, 115, 119],
             '11': [0, 3, 6, 9, 13, 16, 19, 23, 26, 30, 33, 37, 40, 44, 47, 51, 55, 58, 62, 65, 69, 73, 76, 80, 83, 87,
                    90, 94, 98, 101, 105, 108, 112, 116, 119, 123, 127, 130, 134],
             '12': [1, 4, 7, 11, 14, 18, 22, 26, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97,
                    101, 105, 109, 113, 117, 121, 125, 129, 133, 137, 141, 145, 149],
             '13': [1, 4, 8, 12, 16, 20, 24, 28, 33, 37, 41, 45, 50, 54, 59, 63, 67, 72, 76, 80, 85, 89, 94, 98, 102,
                    107, 111, 116, 120, 125, 129, 133, 138, 142, 147, 151, 156, 160, 165],
             '14': [1, 5, 9, 13, 17, 22, 26, 31, 36, 40, 45, 50, 55, 59, 64, 67, 74, 78, 83, 88, 93, 98, 102, 107, 112,
                    117, 122, 127, 131, 136, 141, 146, 151, 156, 161, 165, 170, 175, 180],
             '15': [1, 5, 10, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 70, 75, 80, 85, 90, 96, 101, 106, 111, 117,
                    122, 127, 132, 138, 143, 148, 153, 159, 164, 169, 174, 180, 185, 190, 196],
             '16': [1, 6, 11, 15, 21, 26, 31, 37, 42, 47, 53, 59, 64, 70, 75, 81, 86, 92, 98, 103, 109, 115, 120, 126,
                    132, 137, 143, 149, 154, 160, 166, 171, 177, 183, 188, 194, 200, 206, 211],
             '17': [2, 6, 11, 17, 22, 28, 34, 39, 45, 51, 57, 63, 67, 75, 81, 87, 93, 99, 105, 111, 117, 123, 129, 135,
                    141, 147, 154, 160, 166, 172, 178, 184, 190, 196, 202, 209, 215, 221, 227],
             '18': [2, 7, 12, 18, 24, 30, 36, 42, 48, 55, 61, 67, 74, 80, 86, 93, 99, 106, 112, 119, 125, 132, 138, 145,
                    151, 158, 164, 171, 177, 184, 190, 197, 203, 210, 216, 223, 230, 236, 243],
             '19': [2, 7, 13, 19, 25, 32, 38, 45, 52, 58, 65, 72, 78, 85, 92, 99, 106, 113, 119, 126, 133, 140, 147,
                    154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 258],
             '20': [2, 8, 14, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90, 98, 105, 112, 119, 127, 134, 141, 149, 156,
                    163, 171, 178, 186, 193, 200, 208, 215, 222, 230, 237, 245, 252, 259, 267, 274]
             })

        self.crit_1 = pd.DataFrame({
            '2': [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0,
                  5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 11.0],
            '3': [-1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0, 9.0, 10.0, 11.0,
                  11.0, 12.0, 13.0, 13.0, 14.0, 15.0, 15.0, 16.0, 17.0, 17.0, 18.0, 19.0, 19.0, 20.0, 21.0, 21.0, 22.0,
                  23.0, 23.0, 24.0],
            '4': [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0,
                  18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0,
                  35.0, 36.0, 38.0, 39.0],
            '5': [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30, 32, 33, 35, 36,
                  38, 39, 40, 42, 43, 45, 46, 48, 49, 50, 52, 53],
            '6': [0, 2, 3, 5, 7, 8, 10, 12, 14, 16, 17, 19, 21, 23, 25, 26, 28, 30, 32, 34, 36, 37, 39, 41, 43, 45, 46,
                  48, 50, 52, 54, 56, 57, 59, 61, 63, 65, 67, 68],
            '7': [0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 24, 26, 28, 30, 33, 35, 37, 39, 41, 44, 46, 48, 50, 53, 55, 57,
                  59, 61, 64, 66, 68, 70, 73, 75, 77, 79, 82, 84],
            '8': [1, 3, 5, 8, 10, 13, 15, 18, 20, 23, 26, 28, 31, 33, 36, 39, 41, 44, 47, 49, 52, 54, 57, 60, 62, 65,
                  68, 70, 73, 76, 78, 81, 84, 86, 89, 91, 94, 97, 99],
            '9': [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75,
                  78, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115],
            '10': [1, 4, 7, 11, 14, 17, 20, 24, 27, 31, 34, 37, 41, 44, 48, 51, 55, 58, 62, 65, 68, 72, 75, 79, 82, 86,
                   89, 93, 96, 100, 103, 107, 110, 114, 117, 121, 124, 128, 131],
            '11': [1, 5, 8, 12, 16, 19, 23, 27, 31, 34, 38, 42, 46, 50, 54, 57, 61, 65, 69, 73, 77, 81, 85, 89, 92, 96,
                   100, 104, 108, 112, 116, 120, 124, 128, 131, 135, 139, 143, 147],
            '12': [2, 5, 9, 13, 17, 21, 26, 30, 34, 38, 42, 47, 51, 55, 60, 64, 68, 72, 77, 81, 85, 90, 94, 98, 103,
                   107, 111, 116, 120, 124, 128, 133, 137, 141, 146, 150, 154, 159, 163],
            '13': [2, 6, 10, 15, 19, 24, 28, 33, 37, 42, 47, 51, 56, 61, 65, 70, 75, 80, 84, 89, 94, 98, 103, 108, 113,
                   117, 122, 127, 132, 136, 141, 146, 151, 156, 160, 165, 170, 175, 179],
            '14': [2, 7, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 77, 82, 87, 92, 97, 102, 107, 113, 118,
                   123, 128, 133, 138, 144, 149, 154, 159, 164, 170, 175, 180, 185, 190, 196],
            '15': [3, 7, 12, 18, 23, 28, 33, 39, 44, 50, 55, 61, 66, 72, 77, 83, 88, 94, 100, 105, 111, 116, 122, 128,
                   133, 139, 144, 150, 156, 161, 167, 172, 178, 184, 189, 195, 201, 206, 212],
            '16': [3, 8, 14, 19, 25, 30, 36, 42, 48, 54, 60, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131, 137,
                   143, 149, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228],
            '17': [3, 9, 15, 20, 26, 33, 39, 45, 51, 57, 64, 70, 77, 83, 89, 96, 102, 109, 115, 121, 128, 134, 141, 147,
                   154, 160, 167, 173, 180, 186, 193, 199, 206, 212, 219, 225, 232, 238, 245],
            '18': [4, 9, 16, 22, 28, 35, 41, 48, 55, 61, 68, 75, 82, 88, 95, 102, 109, 116, 123, 130, 136, 143, 150,
                   157, 164, 171, 178, 185, 192, 199, 206, 212, 219, 226, 233, 240, 247, 254, 261],
            '19': [4, 10, 17, 23, 30, 37, 44, 51, 58, 65, 72, 80, 87, 94, 101, 109, 116, 123, 130, 138, 145, 152, 160,
                   167, 174, 182, 189, 196, 204, 211, 218, 226, 233, 241, 248, 255, 263, 270, 278],
            '20': [4, 11, 18, 25, 32, 39, 47, 54, 62, 69, 77, 84, 92, 100, 107, 115, 123, 130, 138, 146, 154, 161, 169,
                   177, 185, 192, 200, 208, 216, 224, 231, 239, 247, 255, 263, 271, 278, 286, 294]
        })

    @staticmethod
    def sampling(data1, data2, decay=0.95):

        n1 = len(data1)
        n2 = len(data2)

        while n1 * n2 > 50000000:
            if n1 >= n2:
                n1 *= decay
            else:
                n2 *= decay
        n1 = np.math.ceil(n1)
        n2 = np.math.ceil(n2)

        if n1 != len(data1):
            data1 = np.random.choice(data1, size=n1, replace=False)
        if n2 != len(data2):
            data2 = np.random.choice(data2, size=n2, replace=False)

        return data1, data2

    def calc_ci(self):

        data1, data2 = self.sampling(self.data1, self.data2)
        N = stats.norm.ppf(1 - self.sig / 2)
        diffs = sorted([i - j for i in data1 for j in data2])
        mid = np.median(diffs)

        # the Kth smallest to the Kth largest of the n x m differences then determine ??? ??? ?? ??
        # the confidence interval, where K is:
        k = np.math.ceil(self.n1 * self.n2 / 2 - (N * (self.n1 * self.n2 * (self.n1 + self.n2 + 1) / 12) ** 0.5))

        return round(diffs[k - 1], 3), round(mid, 3), round(diffs[len(diffs) - k], 3)

    def calc_power(self, n=None, mde=None):
        mu1 = np.mean(self.data1)

        if n:
            effect_size = mu1 * mde
            n1, n2 = n / (np.pi / 3), n / (np.pi / 3)

        else:
            mu2 = np.mean(self.data2)
            effect_size = mu1 - mu2
            n1, n2 = len(self.data1) / (np.pi / 3), len(self.data1) / (np.pi / 3)

        x = np.asarray(self.data1)
        y = np.asarray(self.data2)
        sd = np.std(np.concatenate((x, y)))

        dof = n1 + n2 - 2
        parameter = effect_size / sd / np.sqrt(1 / n1 + 1 / n2)
        critic = stats.t.isf(self.sig / 2, dof)
        greater = stats.nct.sf(critic, dof, parameter)
        less = stats.nct.cdf(-critic, dof, parameter)

        if self.tail == 'two-sided':
            power = greater + less
        elif self.tail == 'greater':
            power = greater
        elif self.tail == 'less':
            power = less

        return power

    def calc_samplesize(self, power=0.8, mde=0.1, available_size=10000000):
        samplesize = optimize.brentq(lambda size: self.calc_power(n=size, mde=mde) - power, 2, available_size)
        samplesize = int(math.ceil(samplesize))
        return samplesize

    def test(self):
        x = np.asarray(self.data1)
        y = np.asarray(self.data2)

        ranked = stats.rankdata(np.concatenate((x, y)))
        rankx = ranked[:self.n1]
        self.u1 = self.n1 * self.n2 + (self.n1 * (self.n1 + 1)) / 2.0 - np.sum(rankx, axis=0)
        self.u2 = self.n1 * self.n2 - self.u1

        self.stat = self.u1 if self.u1 < self.u2 else self.u2
        self.effectsize = 1 - (2 * self.stat) / (self.n1 * self.n2)

        if min(self.n1, self.n2) < 2:
            raise ValueError('data is too small')

        elif 2 <= min(self.n1, self.n2) <= 20 and 2 <= max(self.n1, self.n2) <= 40:

            if self.tail != 'two':
                raise ValueError('data is too small')

            self.sample_size = 'Small'
            self.p = None

            if self.sig == 0.05:
                criticalu = self.crit_05[str(min(self.n1, self.n2))][max(self.n1, self.n2) - 2]
                self.is_sig = self.u <= criticalu

            elif self.sig == 0.1:
                criticalu = self.crit_1[str(min(self.n1, self.n2))][max(self.n1, self.n2) - 2]
                self.is_sig = self.u <= criticalu

        else:

            self.sample_size = 'Large'

            T = stats.tiecorrect(ranked)
            sd = np.sqrt(T * self.n1 * self.n2 * (self.n1 + self.n2 + 1) / 12.0)

            if T == 0:
                raise ValueError('data is too small')
            meanrank = self.n1 * self.n2 / 2.0 + 0.5

            if self.tail == 'two-sided':
                bigu = max(self.u1, self.u2)
            elif self.tail == 'less':
                bigu = self.u1
            elif self.tail == 'greater':
                bigu = self.u2
            z = (bigu - meanrank) / sd

            if self.tail == 'two-sided':
                self.p = 2 * stats.norm.sf(abs(z))
            else:
                self.p = stats.norm.sf(z)

            self.is_sig = self.p <= self.sig

            # ci_result = self.calc_ci()

        return self.stat, self.p