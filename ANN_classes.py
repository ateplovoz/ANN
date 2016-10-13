# -*- coding: utf-8 -*-
# ANN_classes.py
# Классы для нейросети

import numpy as np


# Функция активации
def actfunc(val):
    if val > 0:
        return val
    else:
        return 0.01 * val


# Производная функции активации
def deractfunc(val):
    return (0.01, 1)[val > 0]


class NSignal:
    def __init__(self, inputs):
        self.v = np.ones([inputs])  # Значение
        self.g = np.ones([inputs])  # Градиент
        self.m = np.ones([inputs])  # Обратная инерция


class Neuron:
    def __init__(self, cii):
        self.shake = 0.1  # Величина встряхивания весов
        self.pullback = 0  # Величина упругого возвращения весов
        self.deltaT = 0.01  # Скорость обучения
        self.expscale = 1  # Крутизна экспоненты
        self.Vii = NSignal(cii)  # Массив входов
        self.Vw = NSignal(cii)  # Массив взвешенных входов
        self.Viws = NSignal(1)  # Суммированный сигнал
        self.Voo = NSignal(1)  # Массив выходных сигналов
        self.Vt = np.array([0])  # Массив целевых значений

    # Функция ошибки
    # 1-e^(-b(x-t)^2)
    def errfunc(self, t, x):
        return 1 - np.exp(-self.expscale * np.power(x - t, 2))

    # Производная функции ошибки
    # -2b(x-t)e^(-b(x-t)^2)
    def dererrfunc(self, t, x):
        # return -2 * self.expscale\
        #         * (x - t)\
        #         * self.errfunc(t, x)
        return self.errfunc(t, x)

    # Присваивание градиента
    def getgrad(self, tt=float, appr='direct'):
        self.Vt = tt
        if appr == 'direct':  # Прямой перенос по связям
            self.Voo.g = 1 * self.Vt
        if appr == 'target':  # Вычисление расстояния до цели
            # gr = 1 - np.exp(-self.expscale * np.power(self.Voo.v - self.Vt, 2))
            if self.Vt - self.Voo.v > 0:
                self.Voo.g = self.dererrfunc(self.Vt, self.Voo.v)
            else:
                self.Voo.g = -1 * self.dererrfunc(self.Vt, self.Voo.v)

    def forward(self, ii):  # Прямой прогон
        if len(ii) != self.Vii.v.size:
            raise RuntimeError('V-input size mismatch!')
        self.Vii.v = np.array(ii)
        self.Viws.v = np.sum(self.Vii.v * self.Vw.v)
        self.Voo.v = actfunc(self.Viws.v)
        return self.Voo.v

    def backward(self):  # Обратный прогон
        # Получение градиента iws
        self.Viws.g = self.Voo.g.sum() * deractfunc(self.Viws.v)
        # Получение градиента весов
        self.Vw.g = self.Vii.v\
            * self.Viws.g
        # Получение градиента входов
        self.Vii.g = self.Vw.v\
            * self.Viws.g
        # self.commit()
        return self.Vii.g

    def commit(self):
        # Упругое возвращение весов к нулю (с учётом инерции)
        self.Vw.g += -self.Vw.v * self.Vw.m * self.pullback
        # Корректировка весов
        self.Vw.v += self.Vw.g * self.deltaT
        # Встряхивание весов
        # self.Vw.v += self.Vw.g \
        #     * self.shake \
        #     * (np.random.random(len(self.Vw.g)) * 2 - 1)

    def wgh_tune(self, val, randval):
        self.Vw.v = np.array([val
                              + (np.random.rand() * 2 - 1)
                              * randval for _ in self.Vw.v])


# Получение листа градиентов как их воспринимает нейрон
def unpackgrad(pack):
    unpacked = np.array([sig.g for sig in pack])
    return unpacked


# Получение листа значений как их воспринимает нейрон
def unpackval(pack):
    unpacked = np.array([np.concatenate((sig.v, [1])) for sig in pack])
    return unpacked


class NNetwork:
    def __init__(self, iorder, lorder, oorder):
        # Входной слой нейронов
        self.Lii = [Neuron(1) for _ in xrange(iorder)]
        # Промежуточный слой нейронов
        self.L1 = [Neuron(iorder + 1) for _ in xrange(lorder)]
        # Выходной слой нейронов
        self.Loo = [Neuron(lorder + 1) for _ in xrange(oorder)]
        # Временная переменная хранения входного вектора
        self.Vii = np.array([])
        # Вектор значений выходов входного слоя
        self.VLii = [NSignal(iorder) for _ in xrange(lorder)]
        # Вектор значений выходов промежуточного слоя
        self.VL1 = [NSignal(lorder) for _ in xrange(oorder)]
        # Вектор значений выходов выходного слоя
        self.VLoo = [NSignal(1) for _ in xrange(oorder)]

    # Установка количества переменных во входном слое
    def cfg_input(self, order):
        self.Lii = [Neuron(order) for _ in self.Lii]

    def cfg_mass(self):
        for neur in self.L1:
            template = np.ones([neur.Vw.m.size])
            template[-1] = 0
            neur.Vw.m = template
        for neur in self.Loo:
            template = np.ones([neur.Vw.m.size])
            template[-1] = 0
            neur.Vw.m = template

    def forward(self, ii):  # Прямой прогон
        self.Vii = np.array(ii)
        for VLii in self.VLii:
            VLii.v = np.array([Neurii.forward(self.Vii)
                               for Neurii in self.Lii])
        for VL1 in self.VL1:
            VL1.v = np.array([NeurL1.forward(v) for NeurL1, v
                              in zip(self.L1, unpackval(self.VLii))])
        for VLoo in self.VLoo:
            VLoo.v = np.array([Neuroo.forward(v) for Neuroo, v
                               in zip(self.Loo, unpackval(self.VL1))])

    def getnetgrad(self, val):  # Запись градиентов в выходные связи
        for neur, val in zip(self.Loo, val):
            neur.getgrad(val, appr='target')

    def backward(self):  # Обратный прогон
        for VL1, neur in zip(self.VL1, self.Loo):
            VL1.g = neur.backward()
        for neur, g in zip(self.L1, unpackgrad(self.VL1).T):
            neur.getgrad(sum(g))
        for VLii, neur in zip(self.VLii, self.L1):
            VLii.g = neur.backward()
        for neur, g in zip(self.Lii, unpackgrad(self.VLii).T):
            neur.getgrad(sum(g))
        for _, neur in zip(self.Vii, self.Lii):
            neur.backward()

    # Запись весов
    def ncommit(self):
        for neur in self.Loo:
            neur.commit()
        for neur in self.L1:
            neur.commit()
        for neur in self.Lii:
            neur.commit()

    def nwgh_reset(self, val):
        for neurii, neurl1, neuroo in zip(self.Lii, self.L1, self.Loo):
            neurii.wgh_tune(val, 0)
            neurl1.wgh_tune(val, 0)
            neuroo.wgh_tune(val, 0)

    def nwgh_randomize(self, offset, scale):
        for neurii in self.Lii:
            neurii.wgh_tune(offset, scale)
        for neurl1 in self.L1:
            neurl1.wgh_tune(offset, scale)
        for neuroo in self.Loo:
            neuroo.wgh_tune(offset, scale)
