import pandas as pd
import csv
import os

rows=[]
d = {'keta' : [] , 'sum' : [], 'road': []}
with open('center_data/centrlize.csv', 'r') as csvFile:
    data = csv.reader(csvFile)
    for row in data:
        d['keta'].append(row[0])
        if row[1] == 'nefah':
          d['sum'].append(row[1])
        else:
          d['sum'].append(float(row[1]))
        d['road'].append(row[2])
    df = pd.DataFrame(d)
    sums = df.groupby(['keta','road'])['sum'].sum().reset_index()

    arr1 = sums['sum'].values
    arr2 = sums['keta'].values
    arr3 = sums['road'].values
    arr = []
    for j in range(len(arr1)-1):
        arr.append([arr1[j],arr2[j],arr3[j]])

    arr.sort(key=lambda tup: tup[0])

    new_data = []
    new_data.append(['keta','road','center'])
    i = 0
    # print(arr.__len__())
    for row in arr:
        if i > 420 :
            new_data.append((row[1],row[2],1))
        else:
            new_data.append((row[1],row[2], 0))
        i = i + 1

    with open('output_centerlizeToRoad_tmp.csv', 'w') as csvFile1:
        writer = csv.writer(csvFile1)
        writer.writerows(new_data)
    csvFile1.close()

    with open('output_centerlizeToRoad_tmp.csv') as in_file:
        with open('output_centerlizeToRoad.csv', 'w', newline='') as out_file:
            writer = csv.writer(out_file)
            for row in csv.reader(in_file):
                if any(row):
                    writer.writerow(row)

a = pd.read_csv("output_centerlizeToRoad.csv")
b = pd.read_csv("center_data/xyToRoad.csv")
merged = a.merge(b, on=['keta','road'])
merged.drop(['keta','road'], axis=1,inplace=True)
merged.to_csv("centerlizeToRoad.csv", index=False)
os.remove('output_centerlizeToRoad_tmp.csv')
os.remove('output_centerlizeToRoad.csv')
