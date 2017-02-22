# -*- coding: utf-8 -*-
# ANN_classes.py
# Классы для нейросети
# Переписано для python 3.6

import numpy as np
version = '0.10.0'

class ListSizeError(ValueError):
    pass


class NSignal:
    u"""Нейросигнал, содержит в себе значение, градиент и инерцию"""
    def __init__(self, inputs):
        self.v = np.ones([inputs])  # Значение
        self.g = np.ones([inputs])  # Градиент
        self.m = np.ones([inputs])  # Обратная инерция


class Neuron:
    u"""
    Нейрон, имеет cii входов и один выход
    Внутри себя связан нейросигналами (NSignal)
    """
    def __init__(self, cii):
        self.shake = 0.1  # Величина встряхивания весов
        self.pullback = 0.2  # Величина упругого возвращения весов
        self.deltaT = 0.01  # Скорость обучения
        self.expscale = 1  # Крутизна экспоненты
        self.type = 'relu'  # Тип используемой нелинейности
        self.Vii = NSignal(cii)  # Массив входов
        self.Vw = NSignal(cii)  # Массив взвешенных входов
        self.Viws = NSignal(1)  # Суммированный сигнал
        self.Voo = NSignal(1)  # Массив выходных сигналов
        self.Vt = np.array([0])  # Массив целевых значений

    def errfunc(self, t, x):
        u"""Функция ошибки, возвращает 1-e^(-b(x-t)^2)"""
        return 1 - np.exp(-self.expscale * np.power(x - t, 2))

    def dererrfunc(self, t, x):
        u"""
        Производная функции ошибки, возвращает -2b(x-t)e^(-b(x-t)^2)
        На данный момент возвращает функцию ошибки
        """
        # return -2 * self.expscale\
        #         * (x - t)\
        #         * self.errfunc(t, x)
        return self.errfunc(t, x)

    def actfunc(self, val, type='relu'):
        u""""
        Функция активации
        :val: входное значение
        :type: ('relu'/любой) выбор нелинейности"""
        # if val > 0:
        #     return val
        # elif type=='relu':  # Ректифицированная нелинейность
        #     # return 0.01 * val
        #     # return 0.5 * val
        #     return 0
        if self.type == 'relu':
            return max(0, val)
        else:
            return val  # Без изменения


    def deractfunc(self, val):
        u""""Производная функции активации"""
        # return (0, 1)[val > 0]
        return 1

    def getgrad(self, tt=float, appr='direct'):
        u"""Присваивает внешние градиенты выходным сигналам"""
        self.Vt = tt
        if appr == 'direct':  # Прямой перенос по связям
            self.Voo.g = 1 * self.Vt
            self.pullback = 0.1 * abs(self.Vt)
        if appr == 'target':  # Вычисление расстояния до цели
            # gr = 1 - np.exp(-self.expscale * np.power(self.Voo.v - self.Vt, 2))
            if self.Vt - self.Voo.v > 0:
                self.Voo.g = self.dererrfunc(self.Vt, self.Voo.v)
            else:
                self.Voo.g = -1 * self.dererrfunc(self.Vt, self.Voo.v)
            self.pullback = 0.1 * abs(self.dererrfunc(self.Vt, self.Voo.v))

    def forward(self, ii):
        u"""Рассчитывает и возвращает выход нейрона"""
        if len(ii) != self.Vii.v.size:
            raise ListSizeError('V-input size mismatch!')
        self.Vii.v = np.array(ii)
        self.Viws.v = np.sum(self.Vii.v * self.Vw.v)
        self.Voo.v = self.actfunc(self.Viws.v, self.type)
        return self.Voo.v

    def backward(self):
        u"""рассчитывает градиенты на всех сигналах"""
        # Получение градиента iws
        self.Viws.g = self.Voo.g.sum() * self.deractfunc(self.Viws.v)
        # Получение градиента весов
        self.Vw.g = self.Vii.v * self.Viws.g
        # Получение градиента входов
        self.Vii.g = self.Vw.v * self.Viws.g
        return self.Vii.g

    def commit(self):
        u"""Корректирует значения весов"""
        # Корректировка весов
        self.Vw.v += self.Vw.g * self.deltaT
        # Упругое возвращение весов к нулю (с учётом инерции)
        self.Vw.v += -self.Vw.v * self.Vw.m * self.pullback
        # Встряхивание весов (эл-ты с нулевой инерцией невосприимчивы"
        # self.Vw.v += self.Vw.g \
        #     * self.shake \
        #     * (np.random.random(len(self.Vw.g)) * 2 - 1)

    def wgh_tune(self, val, randval):
        u"""
        Задаёт весам определённое значение
        val - постоянный уровень
        randval - диаметр случайного значения
        """
        self.Vw.v = np.array([val + (np.random.rand() * 2 - 1) * randval for _ in self.Vw.v])


def unpackgrad(vector):
    u"""
    Вытаскивает градиенты из слоя сигналов
    Применяется в основном к промежуточным слоям сигналов
    :param vector: - вектор NSignal (например, соединяющий слой)
    :return:
    """
    unpacked = np.array([sig.g for sig in vector])
    return unpacked


def unpackval(vector):
    u"""
    Вытаскивает значения из слоя сигналов
    Применяется в основном к промежуточным слоям сигналов
    :param vector: - вектор NSignal (например, промежуточный слой)
    """
    # unpackval не отображает фиксированную единицу в слоях, поскольку
    #   она прописана в самой сети
    unpacked = np.array([np.concatenate((sig.v, [1])) for sig in pack])
    return unpacked


class NNetwork:
    u"""
    Нейросеть
    Содержит в себе слой нейронов:
    iorder - кол-во нейронов во входном слое
    lorder - кол-во нейронов в скрытом слое
    oorder - кол-во нейронов в выходном слое
    Между слоями нейронов связана векторами нейросигналов:
    VLii - между входным и скрытым, lorder сигналов по iorder значений
    VL1 - между скрытым и выходным, oorder сигналов по lorder значений
    VLoo - за выходным, oorder сигналов по oorder значений
    """
    def __init__(self, iorder, lorder, oorder):
        self.Lii = [Neuron(1) for _ in xrange(iorder)]  # Входной слой
        self.L1 = [Neuron(iorder + 1) for _ in xrange(lorder)]  # Скрытый слой
        self.Loo = [Neuron(lorder + 1) for _ in xrange(oorder)]  # Выходной слой
        selfVii = np.array([])  # Слой для временного хранения входных данных
        self.VLii = [NSignal(iorder) for _ in xrange(lorder)]  # Вектор входного слоя
        self.VL1 = [NSignal(lorder) for _ in xrange(oorder)]  # Вектор скрытого слоя
        self.VLoo = [NSignal(1) for _ in xrange(oorder)]  # Вектор выходного слоя

    def cfg_input(self, order):
        u"""Установка количества переменных во входном слое"""
        self.Lii = [Neuron(order) for _ in self.Lii]

    def cfg_type(self):
        u"""
        Устанавливает отсутствие нелинейности на некоторых нейронах
        """
        for neur in self.Lii:
            neur.type = 'linear'
        for neur in self.Loo:
            neur.type = 'linear'

    def cfg_mass(self):
        u"""
        Задаёт нулевую инерцию свободным членам суммы
        обязателен к выполнению
        """
        for neur in self.L1:
            template = np.ones([neur.Vw.m.size])
            template[-1] = 0
            neur.Vw.m = template
        for neur in self.Loo:
            template = np.ones([neur.Vw.m.size])
            template[-1] = 0
            neur.Vw.m = template

    def forward(self, ii):  # Прямой прогон
        u"""
        Прямой расчёт сети по слоям нейронов
        Записывает все рассчитанные значения
        Выдаёт результат расчёта сети
        """
        self.Vii = np.array(ii)
        for VLii in self.VLii:
            VLii.v = np.array([Neurii.forward(self.Vii) for Neurii in self.Lii])
        for VL1 in self.VL1:
            VL1.v = np.array([NeurL1.forward(v) for NeurL1, v in zip(self.L1, unpackval(self.VLii))])
        for VLoo in self.VLoo:
            VLoo.v = np.array([Neuroo.forward(v) for Neuroo, v in zip(self.Loo, unpackval(self.VL1))])
        return unpackval(self.VLoo)

    def getnetgrad(self, val):  # Запись градиентов в выходные связи
        u"""Записывает выходные градиенты в нейроны выходного слоя"""
        for neur, val in zip(self.Loo, val):
            neur.getgrad(val, appr='target')

    def backward(self):  # Обратный прогон
        u"""
        Обратный расчёт сети по слоям нейронов
        Рассчитывает и записывает все градиенты
        Ничего не возвращает
        """
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
        u"""Производит корректировку весов"""
        for neur in self.Loo:
            neur.commit()
        for neur in self.L1:
            neur.commit()
        for neur in self.Lii:
            neur.commit()

    def nwgh_reset(self, val):
        u"""Устанавливает все веса нейронов на одно значение"""
        for neurii in self.Lii:
            neurii.wgh_tune(val, 0)
        for neurl1 in self.L1:
            neurl1.wgh_tune(val, 0)
        for neuroo in self.Loo:
            neuroo.wgh_tune(val, 0)

    def nwgh_randomize(self, offset, scale):
        u"""Устанавливает все веса нейронов в диапазоне одного значения: offset +- scale"""
        for neurii in self.Lii:
            neurii.wgh_tune(offset, scale)
        for neurl1 in self.L1:
            neurl1.wgh_tune(offset, scale)
        for neuroo in self.Loo:
            neuroo.wgh_tune(offset, scale)
