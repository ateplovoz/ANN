import csv

data = []

with open('data_m.csv') as datafile:
    reader = csv.reader(datafile)
    for row in reader:
        data.append(row)
