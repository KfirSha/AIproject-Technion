import numpy as np
import pandas as pd
import csv
from dates import Dates
from sklearn import neighbors
from sklearn.neighbors import KDTree
import os

class Data:
  def __init__(self, coordinates, radius):
    self.coordinates = coordinates
    self.radius = radius
    self.dates = Dates(coordinates)
    print("before_int")
    self.dates.make_table_with_coordinates_and_dates()
    print("finish_init")

  def add_feature_to_coordinates(self,feature,type):
      global Final
      m = 0
      for file in feature:
          X = pd.read_csv('features/'+file+'.csv')
          X = X.values
          Y = pd.read_csv("junctions.csv")
          Y = Y.values
          New = []
          New.append(['x','y',file])
          if type == 'count':
            ball_tree = neighbors.BallTree(X, leaf_size=2)
          if type == 'sum' or type == 'avg':
            Z = pd.read_csv('features/' + file + '.csv')
            Z.drop(Z.columns[[2]], axis=1,inplace=True)
            ball_tree = neighbors.BallTree(Z, leaf_size=2)
          x = [Y[0][0], Y[0][1], 0]
          for i in range(len(Y)):
              if type == 'count' or type == 'avg':
                count = ball_tree.query_radius([Y[i]], r=self.radius * 1000, count_only=True)
                x = [Y[i][0], Y[i][1], int(count)]
              if type == 'sum' or type == 'avg':
                  sum = 0
                  ind = ball_tree.query_radius([Y[i]], r=self.radius * 1000)
                  for j in range(len(ind[0])):
                      sum = sum + X[j][2]
                  if type == 'sum':
                    x = [Y[i][0], Y[i][1], sum]
                  if type == 'avg':
                    if int(count) == 0 :
                        x = [Y[i][0], Y[i][1], 0]
                    else:
                        x = [Y[i][0], Y[i][1], sum/int(count)]

              New.append(x)
          # New = self.normalize(New)
          with open("out" + file + '.csv' , "w", newline="") as f:
              writer = csv.writer(f)
              writer.writerows(New)
          b = pd.read_csv("out" + file + '.csv')
          Final=Final.merge(b, on=('x','y'))
          x, y = Final[file].min(), Final[file].max()
          Final[file] = round((Final[file] - x) / (y - x), 3)
          os.remove("out" + file + '.csv')
          m = m + 1

#make csv file "AccRadius.csv" that fits for all junctions their accident nodes within radius
  def fits_coordinate_to_accident_nodes_within_its_radius(self,police_data):
      output = []
      title = ['x', 'y', 'year', 'month', 'day', 'humra']
      output.append(title)
      for fileAcc in police_data:
          X = pd.read_csv('acc/' + fileAcc + '.csv')
          C = pd.read_csv('acc/' + fileAcc + '.csv')
          C.drop(C.loc[:, 'pk_teuna_fikt':'STATUS_IGUN'].columns, axis=1, inplace=True)
          C['X'].replace('', np.nan, inplace=True)
          C.dropna(subset=['X'], inplace=True)
          C = C.values
          X.drop(X.loc[:, 'sug_tik':'YEHIDA'].columns, axis=1, inplace=True)
          X.drop(X.loc[:, 'SHAA':'RAMZOR'].columns, axis=1, inplace=True)
          X.drop(X.loc[:, 'SUG_TEUNA':'STATUS_IGUN'].columns, axis=1, inplace=True)
          X['X'].replace('', np.nan, inplace=True)
          X.dropna(subset=['X'], inplace=True)
          X = X.values
          Y = pd.read_csv("junctions.csv")
          Y = Y.values
          ball_tree = neighbors.BallTree(C, leaf_size=2)
          for i in range(len(Y)):
              ind = ball_tree.query_radius([Y[i]], r=1000)
              for j in (ind[0]):
                  x = [Y[i][0], Y[i][1], int(X[j][1]), int(X[j][2]), int(X[j][3]), int(4-X[j][4])]
                  output.append(x)
      with open("AccRadius.csv", "w", newline="") as f:
          writer = csv.writer(f)
          writer.writerows(output)

  def calculate_classification(self):
      d = {'x': [], 'y': [], 'year': [], 'month': [], 'day': [], 'humra': []}
      with open('AccRadius.csv', 'r') as csvFile:
          data = csv.reader(csvFile)
          for row in data:
              if row[0] == 'x':
                d['x'].append(row[0])
              else:
                d['x'].append(float(row[0]))
              if row[1] == 'y':
                d['y'].append(row[1])
              else:
                d['y'].append(float(row[1]))
              d['year'].append(row[2])
              d['month'].append(row[3])
              d['day'].append(row[4])
              if row[5] == 'humra':
                  d['humra'].append(row[5])
              else:
                  d['humra'].append(float(row[5]))
          df = pd.DataFrame(d)
          sum_of_humra = df.groupby(['x', 'y', 'year', 'month', 'day'])['humra'].sum().reset_index()
          count_of_humra = df.groupby(['x', 'y', 'year', 'month', 'day'])['humra'].count().reset_index()
          one = int(1)
          df1 = df.groupby(['x', 'y', 'year', 'month', 'day'])['humra'].apply(lambda x: (x == one).sum()).reset_index(name='count1')
          two = int(2)
          df2 = df.groupby(['x', 'y', 'year', 'month', 'day'])['humra'].apply(lambda x: (x == two).sum()).reset_index(name='count2')
          three = int(3)
          df3 = df.groupby(['x', 'y', 'year', 'month', 'day'])['humra'].apply(lambda x: (x == three).sum()).reset_index(name='count3')

          s = sum_of_humra['humra'].values
          c = count_of_humra['humra'].values
          x = sum_of_humra['x'].values
          y = sum_of_humra['y'].values
          year = sum_of_humra['year'].values
          month = sum_of_humra['month'].values
          day = sum_of_humra['day'].values
          c1 = df1['count1']
          c2 = df2['count2']
          c3 = df3['count3']

          arr = []
          arr.append(['x', 'y', 'day', 'month', 'year', 'sum_humra', 'count_humra', 'c1','c2','c3'])
          for j in range(len(s)-1):
              arr.append([x[j], y[j], day[j], month[j], float(year[j]), s[j], c[j], c1[j], c2[j], c3[j]])

          with open('data_temp.csv', 'w') as csvFile1:
              writer = csv.writer(csvFile1)
              writer.writerows(arr)
          csvFile1.close()

          with open('data_temp.csv') as in_file:
              with open('data.csv', 'w', newline='') as out_file:
                  writer = csv.writer(out_file)
                  for row in csv.reader(in_file):
                      if any(row):
                          writer.writerow(row)
          os.remove('data_temp.csv')
          print("finish calculate_classification")

  def find_the_most_closest_junction(self,file,col_name,num):
      New = []
      x = ['x', 'y']
      for j in range(0,col_name.__len__()):
          x.append(col_name[j])
      New.append(x)
      X = pd.read_csv(file)
      C = pd.read_csv(file)
      for i in range(0,col_name.__len__()):
        C.drop(col_name[i], axis=1, inplace=True)
      X = X.values
      Y = pd.read_csv("junctions.csv")
      Y = Y.values
      kdt = KDTree(C, leaf_size=30, metric='euclidean')
      for i in range(len(Y)):
          # print(i)
          result = kdt.query([Y[i]], k=num, return_distance=False)
          for m in range(0, num):
              Yind = result[0][m]
              x = [Y[i][0], Y[i][1]]
              for f in range(0, len(X[Yind]) - 2):
                  feature = X[Yind][f]
                  x.append(float(feature))
              New.append(x)
      return New

  def fit_centrelized_junctions_to_table(self):
      self.find_min_and_merge_to_big_table('false',1,'centerlizeToRoad.csv', ['center'], ('x', 'y'))
      print("finish fit_centrelized_junctions_to_table")

  def fit_traffic_signals_junctions_to_table(self):
      global Final
      b = pd.read_csv('traffic_signals.csv')
      # self.merge_files_to_big_table('traffic_signals.csv')
      Final = Final.merge(b, on=('x', 'y'),how='left')
      print("finish fit_traffic_signals_junctions_to_table")

  def merge_files_to_big_table(self,file,fields = ('x','y')):
      b = pd.read_csv(file)
      a = pd.read_csv('data.csv')
      merge_file = a.merge(b, on=fields,how ="left")
      merge_file.to_csv('data.csv', index=False)

  def fit_ages_to_table(self):
      self.find_min_and_merge_to_big_table('true',3,'gil.csv', ['Elders','Adults','Teen'], ('x', 'y'))
      print("finish fit_ages_to_table")

  def fit_ped_to_table(self):
      self.find_min_and_merge_to_big_table('true',1,'ped.csv',['ped','year'], ('x','y','year'),10)
      print("finish fit_ped_to_table")

  def fit_weather(self):
      fields_to_merge = ('x','y','day','month','year')
      self.find_min_and_merge_to_big_table('true',1,'Weather/rain.csv',['rain','day','month','year'],fields_to_merge,3650)
      print("finish rain")
      self.find_min_and_merge_to_big_table('true',1,'Weather/wind.csv',['speed','day','month','year'],fields_to_merge,3650)
      print("finish wind")
      self.find_min_and_merge_to_big_table('true',2,'Weather/temp.csv',['MAX_TEMP','MIN_TEMP','day','month','year'],fields_to_merge,3650)
      print("finish temp")
  def find_min_and_merge_to_big_table(self,normelize,j,file,col_names,fields_to_merge,num = 1):
      global Final
      new_data = self.find_the_most_closest_junction(file, col_names,num)
      with open("tmp.csv", "w", newline="") as f:
          writer = csv.writer(f)
          writer.writerows(new_data)
      b = pd.read_csv("tmp.csv")
      Final = Final.merge(b, on=(fields_to_merge),how='left')
      if normelize=='true':
        for i in range(0,j):
            x,y =Final[col_names[i]].min(),Final[col_names[i]].max()
            Final[col_names[i]]=round((Final[col_names[i]] - x) / (y - x),3)
      os.remove("tmp.csv")

'''
construct Data Class by police data abd fit our junctions to accident nodes. In addition 
calculate the classification for this data
'''
our_data = Data('junctions.csv',10)
print("make class of our data")
police_data = ['H2008104AccData', 'H20091041AccData', 'H20101042AccData', 'H20111041AccData', 'H20121041AccData',
        'H20131041AccData', 'H20141041AccData', 'H20151041AccData', 'H20161041AccData', 'H20171041AccData']
our_data.fits_coordinate_to_accident_nodes_within_its_radius(police_data)
our_data.calculate_classification()

Final = pd.read_csv('data.csv')
'''
merge all dated between 2008-2017 to big table(include holidays:
0 - regular, 1 - shabat , 2 - holiday , 3 - Both)
'''

fields = ('x','y','day','month','year')
dates_with_holiday = pd.read_csv("dates_and_coordinates.csv")
Final = dates_with_holiday.merge(Final, on=fields, how='left')


'''
fill the empty field 'sum_humra'&'count_humra' with zero
'''
'''חומרה 1 - קלה חומרה 2 - בינונית חומרה 3- קשה'''
Final['c1'].replace(np.nan,0, inplace=True)
Final['c2'].replace(np.nan,0, inplace=True)
Final['c3'].replace(np.nan,0, inplace=True)
Final['sum_humra'].replace(np.nan,0, inplace=True)
Final['count_humra'].replace(np.nan,0, inplace=True)
# Final.to_csv('output_our_classification_sum.csv', index=False)
'''
add features that we counts their amount within the radius's junction
'''
features_to_count = ['barandpub','education_co','gas_station_only_israel','industrialarea_co','museum_co','police_stations_co'
                    ,'resturants_onlyCoordinats','shopcenter_co','speedCamera_co','train_co','parking']
our_data.add_feature_to_coordinates(features_to_count,'count')
'''
add features that we sum their amount within the radius's junction
'''
features_to_sum = ['dogs_coordinates']
our_data.add_feature_to_coordinates(features_to_sum,'sum')

'''
add features that we calculate their average within the radius's junction
'''
features_to_avg = ['precentOfAvtala']
our_data.add_feature_to_coordinates(features_to_avg, 'avg')
'''
calculate features by finding the node with minimal distance from our junction
'''
our_data.fit_centrelized_junctions_to_table()
our_data.fit_traffic_signals_junctions_to_table()
our_data.fit_ages_to_table()
our_data.fit_ped_to_table()
'''
calculate Weather Feature - rain , wind and temperature
'''
our_data.fit_weather()
Final.to_csv('data.csv', index=False)
