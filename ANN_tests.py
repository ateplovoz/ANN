# -*- coding: utf-8 -*-
# ANN_tests.py
# Тесты для нейросети

from ANN_classes import NNetwork, NSignal, Neuron
import unittest


class NSignalTest(unittest.TestCase):
    def test_nsignal_correctinit(self):
        u"""Должен инициализировать нейросигнал с 1, 5 и 50 значениями"""
        for t in xrange(50):
            sig = NSignal(t)
            self.assertEqual(t, len(sig.v))

    def test_nsignal_zerofail(self):
        u"""Нейросигнал должен иметь хотя бы одно значение"""
        self.assertRaises(ValueError, NSignal, 0)

    def test_nsignal_negfail(self):
        u"""Нейросигнал не может иметь отрицательное количество значений"""
        self.assertRaises(ValueError, NSignal, -1)

    def test_nsignal_floatfail(self):
        u"""Нейросигнал не может иметь дробное количество значений"""
        self.assertRaises(ValueError, NSignal, 0.5)

    def test_nsignal_typefail(self):
        u"""Не может иметь нечисловое значение входов"""
        self.assertRaises(TypeError, NSignal, 'baka')


class NeuronTest(unittest.TestCase):
    def test_neuron_corrinit(self):
        u"""Должен инициировать нейрон с правильным количеством входных сигналов, одним суммарным и одним выходным"""
        for t in xrange(50):
            neur = Neuron(50)
            self.assertEqual(t, len(neur.Vii.v))
            self.assertEqual(1, len(neur.Viws.v))
            self.assertEqual(1, len(neur.Voo.v))

    def test_neuron_zerofail(self):
        u"""Нейрон должен иметь хотя бы один вход"""
        self.assertRaises(ValueError, Neuron, 0)

    def test_neuron_negfail(self):
        u"""Нейрон не может иметь отрицательное количество входов"""
        self.assertRaises(ValueError, Neuron, -1)

    def test_neuron_floatfail(self):
        u"""Нейрон не может иметь дробное количество входов"""
        self.assertRaises(TypeError, Neuron, 0.5)

    def test_neuron_typefail(self):
        u"""Нейрон не может иметь нечисловое значение входов"""

# TODO: Тест на расчёт математических функций
# TODO: Тест на записывание градиентов после getgrad
# TODO: Тест на прямой расчёт
# TODO: Тест на обратный расчёт
# TODO: Тест на корректировку весов
# TODO: Полный тест на работу нейрона
# TODO: Тест на два последовательных расчёта: вперёд-назад-вперёд-назад-вперёд
# TODO: Тест на тюнинг весов


class NNetworkTest(unittest.TestCase):
    pass

# TODO: Типичные тесты на правильность заполнения аргументов(число, ноль, отрицательный, дробное, нечисловой
# TODO: Тест на иницииализацию векторов
# TODO: Тест на установку входов нейронов входного слоя
# TODO: Тест на задание нулевой инерции постоянной
# TODO: Тест на прямой расчёт
# TODO: Тест на getgrad
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