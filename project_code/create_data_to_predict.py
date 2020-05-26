import numpy as np
import pandas as pd
import csv
from sklearn import neighbors
from sklearn.neighbors import KDTree
import os

class Data:
  def __init__(self, coordinates, radius):
    self.coordinates = coordinates
    self.radius = radius

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
          Final=Final.merge(b, on=('x','y'), how="left")
          x, y = Final[file].min(), Final[file].max()
          Final[file] = round((Final[file] - x) / (y - x), 3)
          os.remove("out" + file + '.csv')
          m = m + 1

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
      C = C.values
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
      a = pd.read_csv('output_our_classification.csv')
      merge_file = a.merge(b, on=fields,how ="left")
      merge_file.to_csv('output_our_classification.csv', index=False)

  def fit_ages_to_table(self):
      self.find_min_and_merge_to_big_table('true',3,'gil.csv', ['Elders','Adults','Teen'], ('x', 'y'))
      print("finish fit_ages_to_table")

  def fit_ped_to_table(self):
      self.find_min_and_merge_to_big_table('true',1,'ped.csv',['ped','year'], ('x','y'),10)
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


our_data = Data('junctions.csv',10)
Final = pd.read_csv('junctions.csv')

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
Final.drop_duplicates(['x','y'],inplace=True)
Final.drop(columns=['year'],inplace=True)
Final.to_csv('data_to_predict.csv', index=False)
