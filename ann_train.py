# -*- coding=utf-8 -*-
# ANN train and test code

import csv

data = np.array([])

with open('data_m.csv') as datafile:
    reader = csv.reader(datafile)
    for row in reader:
        data = np.append(data, row)
data = data.reshape(3,32).astype(float)  # Три 1-тензора длиной 32
data[0] = data[0]*0.001  # Перевод Вт в кВт

# Формируется архив индексов, который потом перемешивается.
# Обучающей, тестовой и валидационной выборке присваиваются одинаковые
# значения. Прирчём соотношения выборок 70:20:10
indx = np.arange(data[0].size)
np.random.shuffle(indx)
n_learn = np.array([])
gt_learn = np.array([])
gv_learn = np.array([])
n_test = np.array([])
gt_test = np.array([])
gv_test = np.array([])
n_val = np.array([])
gt_val = np.array([])
gv_val = np.array([])

for ind in indx[:int(np.floor(indx.size*0.7))]:
    n_learn = np.append(n_learn, data[0][ind])
    gt_learn = np.append(gt_learn, data[1][ind])
    gv_learn = np.append(gv_learn, data[2][ind])
for ind in indx[int(np.floor(indx.size*0.7)):int(np.floor(indx.size*0.9))]:
    n_test = np.append(n_test, data[0][ind])
    gt_test = np.append(gt_test, data[1][ind])
    gv_test = np.append(gv_test, data[2][ind])
for ind in indx[int(np.floor(indx.size*0.9)):]:
    n_val = np.append(n_val, data[0][ind])
    gt_val = np.append(gt_val, data[1][ind])
    gv_val = np.append(gv_val, data[2][ind])

if not(NNeuron):
    print('please run ann class specification script first!')

inputs = [tf.placeholder(tf.float32, name=item) for item in ['gt', 'gv']]
tt = tf.placeholder(tf.float32, name='target')
mln_layout = [10 for _ in range(10)]
mln_tvn = NMLNetwork(inputs, mln_layout, name='tvn')
mln_out = tf.add_n(mln_tvn.get_out())
error = tf.pow(tt - mln_out, 2)

optim = tf.train.MomentumOptimizer()
train.step = optim.minimize(error)


def get_feeder(pick, is_training=False):
    if is_training:
        return form_feeder(inputs + [tt], [gt_learn, gv_learn, n_learn], pick)
    else:
        return form_feeder(inputs + [tt], [gt_test, gv_test, n_learn], pick)


