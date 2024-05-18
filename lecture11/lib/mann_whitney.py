import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy import optimize
import math

class MannWhitney():
    '''
    implemented by JihoKwak.
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