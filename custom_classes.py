# -*- coding: utf-8 -*-
# py3
# Пользовательские классы

import numpy as np
from scipy.stats import chi2, t

# 'major.main.minor-python_version'
verInfo = '0.2.12-py3'


def get_verinfo():
    return verInfo


def check_verinfo(req_ver):
    if verInfo != req_ver:
        print(u'custom_classes version mismatch! requested: {0}, in use: {1}'
              .format(req_ver, verInfo))
    else:
        print(u'custom_classes version check succesful')


class StatWorker():
    u'''
    Класс StatWorker - обработка данных эксперимента
    '''
    def __init__(self):
        self.alpha = 0.05
        self.zstar = 2.326

    def panic(self, msg=''):
        if msg == '':
            raise RuntimeError('StatWorker: something went wrong!')
        else:
            raise RuntimeError('StatWorker: something went wrong! Exactly: {}'
                               .format(msg))

    def zstar_help(self):
        u'''
        Выдаёт таблицу z*
        '''
        print('| alpha |   z*  |\n'
              '|-------|-------|\n'
              '|  99%  | 2.576 |\n'
              '|  95%  | 2.326 |\n'
              '|  95%  | 1.960 |\n'
              '|  90%  | 1.645 |\n')

    def get_params(self):
        print(u'Степень значимости = {0}\nZ* = {1}'
              .format(self.alpha, self.zstar))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_zstar(self, zstar):
        self.zstar = zstar

    def check(self, data):
        u'''
        Проверка на правильный тип массива
        '''
        if type(data) != np.ndarray:
            raise TypeError(
                    'StatWorker: incoming data should be ndarray type!')

    def clean3sigma(self, data, mask=True):
        u'''
        Исключение точек, имеющих отклонение более, чем три сигма
        '''
        self.check(data)
        data_mean = data.mean()
        data_std = data.std()
        data_mask = np.abs(data - data_mean) <= data_std
        if mask:
            return data_mask
        else:
            return data[data_mask]

    def var_interv(self, data):
        u'''
        Функция определения средней точки интервала
        '''
        self.check(data)
        return np.array([
            (data[i] + data[i + 1]) / 2 for i in range(len(data) - 1)])

    def phi(self, u):
        u'''
        Функция phi(u) - нормальное распределение
        '''
        return 1 / np.sqrt(2 * np.pi) * np.power(np.e, -np.square(u) / 2)

    def chi2test(self, data, dtype='data'):
        u'''
        Определение критерия Пирсена для проверки гипотезы о нормлальном
        распределении
        '''
        if dtype == 'data':
            self.check(data)
            # Получаем гистограмму и диапазоны
            (freq, ranges) = np.histogram(data)
        elif dtype == 'hist':
            if dtype(data) != tuple:
                raise TypeError(
                        'StatWorker: incoming data should be tuple type!')
            (freq, ranges) = data
        else:
            (freq, ranges) = (None, None)
            self.panic('chi2test: unknown argument')
        med_var = self.var_interv(ranges)  # Середины интервалов
        h = med_var[1]-med_var[0]
        xav = np.average(med_var, weights=freq)  # "взвешенное" среднее
        # xav = med_var.mean()  # Обычное среднее
        # Взвешенное СКО
        xstd = np.sqrt(np.average(np.square(med_var), weights=freq)
                       - np.square(xav))
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

    def conf_level(self, data, method='std'):
        u'''
        Построение доверительного интервала
        '''
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

        return {
                'res': u'[{0} - {1} - {2}'
                .format(conf_lower, mean, conf_upper),
                'u': conf_upper,
                'l': conf_lower,
                'ul': (conf_upper, conf_lower),
                'rng': conf_range,
                'mean': mean,
                'var': variance,
                'std': std,
                't': tstud
                }

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


# Гидрогазодинамические функции и функции гидрогазодинамики
class GGazWorker():
    def __init__(self):
        self.k_air = 1.4

    def pi_lambda(self, g_lambda):
        u'''
        pi(lambda): расчёт безразмерного давления от безразмерной скорости
        '''
        return pow((1-(self.k_air-1)/(self.k_air+1)
                    * pow(g_lambda, 2)), self.k_air/(self.k_air-1))

    def lambda_pi(self, g_pi):
        u'''
        lambda(pi): расчёт безразмерной скорости от безразмерного давления
        '''
        return pow((1-pow(g_pi, (self.k_air-1)/self.k_air))
                   * (self.k_air+1)/(self.k_air-1), 0.5)

    def q_lambda(self, g_lambda):
        u'''
        q(lambda): расчёт безразмерного расхода от безразмерной скорости
        '''
        return pow((self.k_air+1)/2,
                   1/(self.k_air-1))\
            * g_lambda\
            * pow((1-(self.k_air-1)/(self.k_air+1)*pow(g_lambda, 2)),
                  1/(self.k_air-1))


# Полускрипт на вывод пронумерованных параметров
# def print_numbered(itemlist, str_format=u'{0} - {1}'):
def print_numbered(itemlist):
    u'''
    Выводит лист в виде нумерованного списка
    itemlist — лист.
    '''
    i = 0
    for item in itemlist:
        print(str(i) + '\t' + item)
#        print(str_format.format(i, item)
        i += 1


def Cp_methane(t_start, t_end, afr, use_new=False):
    '''
    Расчёт теплоёмкости продуктов сгорания при сжигании природного газа
    (сметана). Возвращает значение теплоёмкости в [Дж/кг°C]
    t_start, t_end [°C]-- температура газа;
    afr [0 ÷ 1]-- соотношение топлива к воздуху. AFR = 0 -- расчёт теплоёмкости
    воздуха.
    use_new [bool] -- использование новой таблицы коэф. полинома
    '''

    gas_coef = {
        'air': [.252192, -.593306, 11.2026, -76.8453, 276.44, -515.04, 392.2],
        'co2': [.104706, 2.11718, -13.1785, 56.2368, -154.60, 243.69, -166.7],
        'h2o': [.448938, -.544201, 13.4255, -65.9598, 159.88, -192.86, 89.17],
        'o2':  [.208363, -.056140, 7.45289, -68.3167, 292.27, -614.50,  512.0],
        'h':   [3.07088, 11.1537, -163.633, 1330.410, -5513.1, 11419., -9424.],
        }

    gas_coef_new = {
        'air': [1.004117, -2.754852*10**-6, 6.890151*10**-7, -8.358055*10**-10,
                3.343216*10**-13, 7.965546*10**-17, -1.175786*10**-19,
                3.856751*10**-23, -4.344547*10**-27],
        'co2': [0.8148627, 1.104033*10**-3, -1.301332*10**-6, 1.323860*10**-9,
                -1.118083*10**-12, 6.735382*10**-16, -2.561333*10**-19,
                5.420749*10**-23, -4.844547*10**-27],
        'h2o': [1.858979, 2.066147*10**-4, 1.409027*10**-6, -2.616702*10**-9,
                3.558973*10**-12, -3.276883*10**-15, 1.857165*10**-18,
                -6.186433*10**-22, 1.112322*10**-25, -8.334864*10**-30],
        'o2':  [0.9146970, 1.026171*10**-4, 1.142046*10**-6, -2.773659*10**-9,
                3.127974*10**-12, -1.990618*10**-15, 7.322529*10**-19,
                -1.451865*10**-22, 1.200398*10**-26],
        'h':   [14.19732, 3.926051*10**-3, -1.924455*10**-5, 4.817688*10**-8,
                -6.327742*10**-11, 5.072448*10**-14, -2.57018*10**-17,
                8.017741*10**-21, -1.402408*10**-24, 1.050463*10**-28],
            }
    g_coef = {
        'air': 0,
        'co2': 0.75,
        'h2o': 0.25,
        'o2': 0,
        'h': 0
        }

    poly_coef = []
    if use_new:
        t1 = t_start
        t2 = t_end
        ts = (t_start + t_end)/2
        gas_coef = gas_coef_new
    else:
        t1 = (t_start + 273.15)*0.0001
        t2 = (t_end + 273.15)*0.0001
        ts = (t1 + t2)/2.

    g_vozd = 1 - afr  # Количество воздуха в газах
    g_co2 = g_coef['co2']*afr*3.66666666  # Количество СО2 в газах
    g_h2o = g_coef['h2o']*afr*9  # Количество водяного пара в газах

    poly_coef = [
        gas_coef['air'][0]*g_vozd + gas_coef['co2'][0]*g_co2 +
        gas_coef['h2o'][0]*g_h2o,
        gas_coef['air'][1]*g_vozd + gas_coef['co2'][1]*g_co2 +
        gas_coef['h2o'][1]*g_h2o,
        gas_coef['air'][2]*g_vozd + gas_coef['co2'][2]*g_co2 +
        gas_coef['h2o'][2]*g_h2o,
        gas_coef['air'][3]*g_vozd + gas_coef['co2'][3]*g_co2 +
        gas_coef['h2o'][3]*g_h2o,
        gas_coef['air'][4]*g_vozd + gas_coef['co2'][4]*g_co2 +
        gas_coef['h2o'][4]*g_h2o,
        gas_coef['air'][5]*g_vozd + gas_coef['co2'][5]*g_co2 +
        gas_coef['h2o'][5]*g_h2o,
        gas_coef['air'][6]*g_vozd + gas_coef['co2'][6]*g_co2 +
        gas_coef['h2o'][6]*g_h2o
        ]

#    i = 0
#    cp  = 0
#    for poly in poly_coef:
#        cp += poly*pow(t_gas*0.0001*(i+1), i)
#        i += 1

    if use_new:
        cp = 0
        i = 0
        for poly in poly_coef:
            cp += poly*pow(ts, i)*(i+1)
            i += 1
        return cp*1000
    else:
        cp = poly_coef[0] + poly_coef[1]*(t1 + t2)
        i = 2
        for poly in poly_coef[2:]:
            cp += poly*pow(ts, i)*(i+1)
            i += 1
        return cp*4186.8


def print_table(sep='\t|', *args):
    for item_row in args:
        row = ''
        for item_col in item_row:
            if type(item_col) == str:
                row = row + item_col + '{}'.format(sep)
            else:
                row = row + '{0:3.2f}{1}'.format(item_col, sep)
        print(row)


def sqdistance(ar1, ar2, order=1):
    try:  # Проверка на итератор
        iter(ar1)
        iter(ar2)
    except TypeError:
        raise TypeError("input arguments should be iterable!")
    if len(ar1) != len(ar2):  # Проверка на одинаковую длину
        raise TypeError("arguments should be same length!")
    dist = 0
    for item1, item2 in zip(ar1, ar2):
        dist += abs(item1**order - item2**order)
    return dist
