# -*- coding: utf-8 -*-
# Пользовательские классы

import numpy as np
from scipy.stats import chi2, t


# Класс StatWorker - обработка данных эксперимента
class StatWorker():
    def __init__(self):
        self.alpha=0.05
        self.zstar = 2.326

    def panic(self, msg=''):
        if msg == '':
            raise RuntimeError('StatWorker: something went wrong!')
        else:
            raise RuntimeError('StatWorker: something went wrong! Exactly: {}'.format(msg))


    # Выдаёт таблицу z*
    def zstar_help(self):
        print '| alpha |   z*  |\n' \
              '|-------|-------|\n' \
              '|  99%  | 2.576 |\n' \
              '|  95%  | 2.326 |\n' \
              '|  95%  | 1.960 |\n' \
              '|  90%  | 1.645 |\n'

    def get_params(self):
        print u'Степень значимости = {0}\nZ* = {1}'.format(self.alpha, self.zstar)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_zstar(self, zstar):
        self.zstar = zstar

    # Проверка на правильный тип массива
    def check(self, data):
        if type(data) != np.ndarray:
            raise TypeError('StatWorker: incoming data should be ndarray type!')

    # Исключение точек, имеющих отклонение более, чем три сигма
    def clean3sigma(self, data):
        self.check(data)
        data_mean = data.mean()
        data_std = data.std()
        data_mask = np.abs(data - data_mean) < data_std
        return data_mask

    # Функция определения средней точки интервала
    def var_interv(self, data):
        self.check(data)
        return np.array([(data[i] + data[i + 1]) / 2 for i in xrange(len(data) - 1)])

    # Функция phi(u) - нормальное распределение
    def phi(self, u):
        return 1 / np.sqrt(2 * np.pi) * np.power(np.e, -np.square(u) / 2)

    # Определение критерия Пирсена для проверки гипотезы о нормлальном распределении
    def chi2test(self, data, dtype='data'):
        if dtype == 'data':
            self.check(data)
            (freq, ranges) = np.histogram(data)  # Получаем гистограмму и диапазоны
        elif dtype == 'hist':
            if dtype(data) != tuple:
                raise TypeError('StatWorker: incoming data should be tuple type!')
            (freq, ranges) = data
        else:
            (freq, ranges) = (None, None)
            self.panic('chi2test: unknown argument')
        med_var = self.var_interv(ranges)  # Середины интервалов
        h = med_var[1]-med_var[0]
        xav = np.average(med_var, weights=freq)  # "взвешенное" среднее
        # xav = med_var.mean()  # Обычное среднее
        xstd = np.sqrt(np.average(np.square(med_var), weights=freq) - np.square(xav))  # Взвешенное СКО
        # xstd = med_var.std()  # обычное СКО
        ranges_i = freq.sum() * h / xstd * self.phi((med_var - xav) / xstd)
        chi2exp = (np.square(freq - ranges_i) / ranges_i).sum()
        k = med_var.size - 3  # Количество степеней свободы
        chi2Cr = chi2.ppf(1 - self.alpha, k)  # Критическое значение хи-квадрат
        return {
            'freq': freq,
            'ranges': ranges,
            'ranges_i': ranges_i,
            'chi2cr': chi2Cr,
            'chi2': chi2exp,
            'av': xav,
            'std': xstd,
            'H0': chi2exp < chi2Cr
        }

    # Построение доверительного интервала
    def conf_level(self, data, method='std'):
        self.check(data)
        conf_lower = None
        conf_upper = None
        tstud = None
        mean = data.mean()
        std = data.std()
        variance = data.var()
        n = data.size

        if method == 'var':
            # Расчёт по дисперсии
            tstud = t.ppf(1 - self.alpha / 2, n - 1)
            conf_lower = mean - tstud * variance / np.sqrt(n)
            conf_upper = mean + tstud * variance / np.sqrt(n)
        elif method == 'std':
            # Расчёт по СКО
            tstud = t.ppf(1 - self.alpha / 2, n - 1)
            conf_range = tstud * std / np.sqrt(n)
            conf_lower = mean - conf_range
            conf_upper = mean + conf_range

        else:
            self.panic('conf_level: unknown argument')

        return {'res':u'[{0} - {1} - {2}'.format(conf_lower, mean, conf_upper),
                'u': conf_upper,
                'l': conf_lower,
                'ul': (conf_upper, conf_lower),
                'rng': conf_range,
                'mean': mean,
                'var': variance,
                'std': std,
                't': tstud}

    # Исключение точек, не попадающих в доверительный интервал
    def conf_clean(self, data):
        self.check(data)
        mask = np.array([], dtype=bool)
        for i in np.arange(data.size):
            data_miss = np.delete(data, i)
            (cu, cl) = self.conf_level(data_miss)['ul']
            if cl <= data[i] <= cu:
                mask = np.append(mask, [True])
            else:
                mask = np.append(mask, [False])
        return mask



