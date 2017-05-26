ann_from_learnset = mln_sym.calc(learnset, DSOCKET)
csv_dataset = np.array([learnset[item][1] for item in range(7)])

with open('data_ANN_uniform_indexed.csv', 'w') as datafile:
    writer = csv.writer(datafile)
    writer.writerows(csv_learnset)

with open('ANN_data_uniform_indexed.csv', 'w') as datafile:
    writer = csv.writer(datafile)
    writer.writerows(ann_from_learnset)
