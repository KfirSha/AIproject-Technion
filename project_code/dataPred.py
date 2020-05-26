import pandas as pd
import datetime
import heapq
from sklearn.ensemble import ExtraTreesClassifier

sug_yom = 0.0
rain = 0.0
speed = 0.1
MAX_TEM = 0.4
MIN_TEMP = 0.3
number_of_resources = 20
classify_by = 'Total Count'
'''
get the input from user
you can choose the param of classification : Severe / Medium / Slightly / Total Count / Total Sum
'''
# sug_yom = input("Enter 0 for regular day , 1 for Saturday, 2 for holiday, 3 for Saturday&holiday...\n")
# rain = input("Enter amount of rainfall ...\n")
# speed = input("Enter the wind speed ...\n")
# MAX_TEM = input("Enter maximum temperature ...\n")
# MIN_TEMP = input("Enter minimum temperature ...\n")
# number_of_resources = input("Enter your number of resources(police cars) ...\n")
# '''you can choose the param of classification : Severe / Medium / Slightly / Total'''
# classify_by = input("Choose your classification preference:\n"
#                "by total number of accidents press 'Total Count'\n"
#                "by number of slightly accidents press 'Slightly'\n"
#                "by number of medium accidents press 'Medium'\n"
#                "by number of sever accidents press 'Sever'\n"
#                "by total sum of accidents press 'Total Sum'\n")
# '''calculate data for prediction'''
df1 = pd.read_csv("data_to_predict.csv")
df1.fillna(0, inplace=True)
df1['sug_yom'] = sug_yom
df1['rain'] = rain
df1['speed'] = speed
df1['MAX_TEMP'] = MAX_TEM
df1['MIN_TEMP'] = MIN_TEMP
X_pred = df1[['Adults','center','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]
junction = df1[['x','y']]
print("Calculating your result ...will take 5 min ")
df2 = pd.read_csv("data.csv")
df2.fillna(0 , inplace=True)
X=df2[['Adults','center','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]

y = df2['count_humra']
if classify_by == 'Slightly':
    y = df2['c1']
if classify_by == 'Medium':
    y = df2['c2']
if classify_by == 'Severe':
    y = df2['c3']
if classify_by == 'Total Sum':
    y = df2['sum_humra']

clf = ExtraTreesClassifier(class_weight='balanced',n_estimators=13)
clf.fit(X,y)
y_pred=clf.predict(X_pred)
output = []

for i in range(y_pred.size):
    if y_pred[i] > 0:
        output.append(((int(y_pred[i])),junction.iloc[i]['x'],junction.iloc[i]['y']))

# print(output)
danger_junctions = heapq.nlargest(number_of_resources, output)
print(danger_junctions)