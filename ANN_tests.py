# -*- coding: utf-8 -*-
# ANN_tests.py
# Тесты для нейросети

import ANN_classes as ann
from ANN_classes import ListSizeError
import numpy as np
import unittest


class NSignalTest(unittest.TestCase):
    def test_nsignal_correctinit(self):
        u"""Должен инициализировать нейросигнал с 1, 5 и 50 значениями"""
        for t in xrange(50):
            sig = ann.NSignal(t)
            self.assertEqual(t, len(sig.v))

    def test_nsignal_zerofail(self):
        u"""Нейросигнал должен иметь хотя бы одно значение"""
        self.assertRaises(ValueError, ann.NSignal, 0)

    def test_nsignal_negfail(self):
        u"""Нейросигнал не может иметь отрицательное количество значений"""
        self.assertRaises(ValueError, ann.NSignal, -1)

    def test_nsignal_floatfail(self):
        u"""Нейросигнал не может иметь дробное количество значений"""
        self.assertRaises(ValueError, ann.NSignal, 0.5)

    def test_nsignal_typefail(self):
        u"""Не может иметь нечисловое значение входов"""
        self.assertRaises(TypeError, ann.NSignal, 'baka')


class NeuronTest(unittest.TestCase):
    def test_neuron_corrinit(self):
        u"""Должен инициировать нейрон с правильным количеством входных сигналов, одним суммарным и одним выходным"""
        for t in xrange(1, 50):
            neur = ann.Neuron(t)
            self.assertEqual(t, len(neur.Vii.v))
            self.assertEqual(1, len(neur.Viws.v))
            self.assertEqual(1, len(neur.Voo.v))

    def test_neuron_zerofail(self):
        u"""Нейрон должен иметь хотя бы один вход"""
        self.assertRaises(ValueError, ann.Neuron, 0)

    def test_neuron_negfail(self):
        u"""Нейрон не может иметь отрицательное количество входов"""
        self.assertRaises(ValueError, ann.Neuron, -1)

    def test_neuron_floatfail(self):
        u"""Нейрон не может иметь дробное количество входов"""
        self.assertRaises(TypeError, ann.Neuron, 0.5)

    def test_neuron_typefail(self):
        u"""Нейрон не может иметь нечисловое значение входов"""
        self.assertRaises(TypeError, ann.Neuron, 'baka')

    def test_errfunc_notgreaterone(self):
        u"""
        Функция ошибки не должна быть больше единицы или меньше минус единицы
        При положительной ошибке должна выдавать положительное значение
        При отрицательной ошибке должна выдавать отрицательное значение
        """
        neur = ann.Neuron(1)
        [self.assertLessEqual(neur.errfunc(50, x), 1) for x in np.linspace(50.1, 100, 200)]
        [self.assertGreaterEqual(neur.errfunc(50, x), -1) for x in np.linspace(0, 49.9, 200)]

    def test_errfunc_atzeroiszero(self):
        u"""Функция ошибки при нуле должна выдавать ноль"""
        neur = ann.Neuron(1)
        [self.assertEqual(neur.errfunc(t, t), 0) for t in np.linspace(0, 100, 200)]

    def test_getgrad_gradwrite_dir(self):
        u"""Записанный с помощью getgrad градиент должен совпадать с выходным"""
        neur = ann.Neuron(1)
        for t in np.linspace(-100, 100, 200):
            neur.getgrad(t)
            self.assertEqual(t, neur.Voo.g)

    def test_getgrad_gradwrite_target(self):
        u"""Рассчитанная с помощью getgrad ошибка должна совпадать с функцией ошибки"""
        neur = ann.Neuron(1)
        for t in np.linspace(-100, 100, 200):
            neur.Voo.v = t
            neur.getgrad(0, appr='target')
            self.assertEqual(neur.Voo.g, neur.errfunc(0, t))

    def test_forward_pos(self):
        u"""
        После прямого расчёта должно получиться детерминированное значение
        Зависит от результатов теста wgh_tune
        """
        for t in xrange(0, 100):
            neur = ann.Neuron(t)
            neur.wgh_tune(1., 0)
            neur.forward([1.0 for _ in xrange(t)])
            self.assertEqual(neur.Voo.v, t)
        for t in xrange(0, 100):
            pick = np.random.rand()
            neur = ann.Neuron(t)
            neur.wgh_tune(1., 0)
            neur.forward([pick for _ in xrange(t)])
            self.assertEqual(neur.Voo.v, pick * t)

    def test_forward_neg(self):
        u"""
        После прямого расчёта должно получиться детерминированное значение
        Зависит от результатов теста wgh_tune
        При отрицательном входе результат умножается на 0.01
        """
        for t in xrange(0, 100):
            neur = ann.Neuron(t)
            neur.wgh_tune(1., 0)
            neur.forward([-1.0 for _ in xrange(t)])
            self.assertEqual(0.01 * neur.Voo.v, -1 * t)
        for pick in [0, 1., 12.3, -5, -1922.4, 200]:
            neur = ann.Neuron(t)
            neur.wgh_tune(1., 0)
            neur.forward([-1 * pick for _ in xrange(t)])
            self.assertEqual(0.01 * neur.Voo.v, -1 * pick * t)

    def test_backward_pospos(self):
        u"""
        В результате обратного расчёта:
         градиент веса больше нуля при положительном градиенте и положительном входе
         градиент iws должен являться суммой выходных градиентов
         градиент iws должен умножаться на 0.01 при отрицательных значениях iws
        """
        neur = ann.Neuron(1)
        neur.wgh_tune(1., 0)
        neur.forward([1.])
        neur.getgrad(np.array([2.]))
        neur.backward()
        [self.assertGreater(t, 0) for t in neur.Vw.g]

    def test_backward_posneg(self):
        u"""Градиент веса меньше нуля при положительном градиенте и отрицательном входе"""
        neur = ann.Neuron(1)
        neur.wgh_tune(1., 0)
        neur.forward([-1.])
        neur.getgrad(np.array([2.]))
        neur.backward()
        [self.assertLess(t, 0) for t in neur.Vw.g]

    def test_backward_negpos(self):
        u"""Градиент веса меньше нуля при отрицательном градиенте и положительном входе"""
        neur = ann.Neuron(1)
        neur.wgh_tune(1., 0)
        neur.forward([1.])
        neur.getgrad(np.array([0.]))
        neur.backward()
        [self.assertLess(t, 0) for t in neur.Vw.g]

    def test_backward_negneg(self):
        u"""Градиент веса меньше нуля при положительном градиенте и отрицательном входе"""
        neur = ann.Neuron(1)
        neur.wgh_tune(1., 0)
        neur.forward([-1.])
        neur.getgrad(np.array([0.]))
        neur.backward()
        [self.assertGreater(t, 0) for t in neur.Vw.g]

    def test_commit_gradneg(self):
        u"""Вес должен уменьшится при отрицательном градиенте"""
        neur = ann.Neuron(1)
        neur.wgh_tune(1., 0)
        old_wgh = neur.Vw.v.tolist()
        neur.forward([1.])
        neur.getgrad(np.array([0.]))
        neur.backward()
        neur.commit()
        self.assertLess(neur.Vw.v.tolist(), old_wgh)

    def test_commit_gradpos(self):
        u"""Вес должен уменьшится при отрицательном градиенте"""
        neur = ann.Neuron(1.)
        neur.wgh_tune(1., 0)
        old_wgh = neur.Vw.v.tolist()
        neur.forward([1.])
        neur.getgrad(np.array([2.]))
        neur.backward()
        neur.commit()
        self.assertGreater(neur.Vw.v.tolist(), old_wgh)

# TODO: Полный тест на работу нейрона

# TODO: Тест на два последовательных расчёта: вперёд-назад-вперёд-назад-вперёд

    def test_wgh_tune_val(self):
        u"""Первый аргумент должен устанавливать определённое значение"""
        neur = ann.Neuron(1)
        for pick in [0, 1., 12.3, -5, -1922.4, 200]:
            neur.wgh_tune(pick, 0)
            self.assertEqual(neur.Vw.v, [pick])

    def test_wgh_tune(self):
        u"""Второй аргумент должен рандомизировать вес около определённого значения"""
        neur = ann.Neuron(1)
        for pick in [0, 1., 12.3, -5, -1922.4, 200]:
            neur.wgh_tune(0, pick)
            self.assertNotEqual(neur.Vw.v, [pick])
            self.assertTrue(0 - pick / 2 < neur.Vw.v[0] < 0 + pick / 2)

    def test_wronglistlength(self):
        u"""forward должен принимать ровно столько входных значений, сколько у него входов"""
        for t in xrange(1, 10):
            neur = ann.Neuron(t)
            self.assertRaises(ListSizeError, neur.forward, np.ones([t + 1]))
            self.assertRaises(ListSizeError, neur.forward, np.ones([t - 1]))

    def test_wronglisttype(self):
        u"""forward должен принимать только ndarray листы"""
        neur = ann.Neuron(1)
        self.assertRaises(TypeError, neur.forward, [1])


class NNetworkTest(unittest.TestCase):
    pass

# TODO: Типичные тесты на правильность заполнения аргументов(число, ноль, отрицательный, дробное, нечисловой

    def test_Lsizes(self):
        u"""Нейросеть должна содержать указанное количество нейронов в слоях"""
        a = 5
        b = 10
        c = 15
        nw = ann.NNetwork(a, b, c)
        self.assertEqual(a, len(nw.Lii))
        self.assertEqual(b, len(nw.L1))
        self.assertEqual(c, len(nw.Loo))

    def test_VLsizes(self):
        u"""Нейросеть должна содержать вектора соответсвующей длины"""
        for _ in xrange(50):
            a, b, c = np.random.randint(1, 10, 3)
            nw = ann.NNetwork(a, b, c)
            self.assertEqual(b, len(nw.VLii), 'expected: {}'.format(b))
            self.assertEqual(a, len(nw.VLii[0].v), 'expected: {}'.format(a))
            self.assertEqual(c, len(nw.VL1), 'expected: {}'.format(c))
            self.assertEqual(b, len(nw.VL1[0].v), 'expected: {}'.format(b))
            self.assertEqual(c, len(nw.VLoo), 'expected: {}'.format(c))
            self.assertEqual(1, len(nw.VLoo[0].v), 'expected: {}'.format(1))

    def test_cfginput(self):
        u"""cfg_input должен задавать указанное количество входов всем нейронам во входном слое"""
        nw = ann.NNetwork(5, 6, 7)
        nw.cfg_input(5)
        for neur in nw.Lii:
            self.assertEqual(5, len(neur.Vii.v))

    def test_cfgmass(self):
        u"""Постоянный член (единица) должен иметь нулевую инерцию, чтобы быть невосприимчивым к возвращению весов"""
        for _ in xrange(50):
            a, b, c = np.random.randint(1, 10, 3)
            nw = ann.NNetwork(a, b, c)
            nw.cfg_mass()
            for neur in nw.L1:
                self.assertEqual(0, neur.Vw.v[-1])
            for neur in nw.Loo:
                self.assertEqual(0, neur.Vw.v[-1])

# TODO: Тест на прямой расчёт
    def test_forward(self):
        u"""
        Прямой расчёт должен выдавать детермированное значение
        Зависит от теста на wgh_set
        """
        nw1 = ann.NNetwork(1, 2, 1)
        nw1.nwgh_reset(1)
        res = nw1.forward([10])
        self.assertEqual(23, res)
        nw2 = ann.NNetwork(1, 2, 1)
        nw2.nwgh_reset(2.5)
        res = nw2.forward([2.2])
        self.assertEqual(83.75, res)

# TODO: Тест на getgrad
    def test_ngetgrad(self):
        a, b, c = np.random.randint(1, 10, 3)
        nw = ann.NNetwork(a, b, c)

# TODO: Тест на обратный расчёт
# TODO: Тест на корректировку весов
# TODO: Тест на установку весов в слоях
# TODO: Тест на рандомизацию весов в слоях
# TODO: Тест на полный расчёт сети вперёд-назад-вперёд
# TODO: Тест на полный расчёт сети вперёд-назад-вперёд-назад-вперёд
# TODO: Тест на распаковку значений вектора
# TODO: Тест на распаковку градиентов вектора

if __name__ == '__main__':
    unittest.main()
