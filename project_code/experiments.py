import numpy as np
import pandas as pd
from multiscorer.multiscorer import MultiScorer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, mean_squared_error,r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import random
from statistics import mean
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn import linear_model
from imblearn.under_sampling import RandomUnderSampler,TomekLinks,NearMiss
from imblearn.over_sampling import SMOTE
import pickle
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = list(unique_labels(y_true, y_pred))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # ax.tight_layout()
    plt.xlim(-0.5, 5.5)
    plt.ylim(5.5, -0.5)
    return ax
def f1_calc(y,y_pred):
    # plot_confusion_matrix(y, y_pred, classes=[0, 1, 2, 3, 4, 5], normalize=True,
    #                       title='Normalized confusion matrix')
    # print(f1_score(y,y_pred,average=None))
    return f1_score(y,y_pred,average='micro')
def matthews(y,y_pred):
    return(matthews_corrcoef(y,y_pred))
def confuse(y,y_pred):
    print(f1_score(y,y_pred,average=None))
def calc_err(y,y_pred):
    error_ = y - y_pred
    return error_.sum() / len(y)
'''start here'''
df = pd.read_csv("data.csv")
df.fillna(0 , inplace=True)
'''make the tests on 5 years'''
df1 = df[df['year'] == 2010]
df2 = df[df['year'] == 2012]
df = pd.concat([df1,df2],ignore_index=True)
df3 = df[df['year'] == 2014]
df = pd.concat([df,df3],ignore_index=True)
df3 = df[df['year'] == 2015]
df = pd.concat([df,df3],ignore_index=True)
df3 = df[df['year'] == 2016]
# df = pd.concat([df,df3],ignore_index=True)
X=df[['trafiic_signal','Elders','Adults','Teen','precentOfAvtala','center','train_co','parking',
      'dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co',
      'museum_co','police_stations_co','barandpub','education_co','gas_station_only_israel','sug_yom',
      'ped','rain','speed','MAX_TEMP','MIN_TEMP']]
X_new=df[['Adults','Teen','precentOfAvtala','center','train_co','parking','dogs_coordinates',
          'resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','museum_co',
          'police_stations_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped',
          'rain','speed','MAX_TEMP','MIN_TEMP']]
X_temp=df[['rain','speed','MAX_TEMP','MIN_TEMP']]
y=df['count_humra']
X_train_All,X_test_ALL,y_train_ALL,y_test_ALL=train_test_split(X,y,test_size=.3,random_state=0,shuffle=False)
# X_train,X_test,y_train,y_test=train_test_split(X_new,y,test_size=.3,random_state=0,shuffle=False)
X_train_temp,X_test_temp,y_train_temp,y_test_temp=train_test_split(X_temp,y,test_size=.3,random_state=0,shuffle=False)
scorer = MultiScorer({
  'R2-Score': (r2_score, {}),
  'MSE': (mean_squared_error, {}),
  'error': (calc_err, {})
})
scorer2 = MultiScorer({
  'eroor': (calc_err, {}),
  'accuracy_score': (accuracy_score, {})
})
scorer3 = MultiScorer({
  'error': (calc_err, {}),
  'f1_calc': (f1_calc, {}),
  'matthews': (matthews, {})
})
'''---------------------------------REGRESSORS-TREES-TEST1------------------------------------------------'''
Regs={"DecisionTreeRegressor" : DecisionTreeRegressor(random_state=0),
      "RandomForestRegressor" : RandomForestRegressor(n_estimators=10,random_state=0),
      "ExtraTreesRegressor" : ExtraTreesRegressor(n_estimators=10,random_state=0)}
for reg in Regs :
    print(reg)
    cross_val_score(Regs.get(reg), X_new, y, scoring=scorer, cv=KFold(n_splits=5))
    results = scorer.get_results()
    for metric in results.keys():
      print("%s: %.3f" % (metric, np.average(results[metric])))
'''----------------------------------------------------------------------------------------------------------------'''
'''-------------------------------CLASSIFIERS TREES TEST2--------------------------------------------'''
'''------DecisitionTreeClassifier----'''
print("DecisitionTreeClassifier-parameters")
parameters= [2,50,200,None]
result=[]
for K in parameters:
    print("max_depth")
    print(K)
    clf = DecisionTreeClassifier(max_depth=K,random_state=0)
    cross_val_score(clf, X, y, scoring=scorer3, cv=KFold(n_splits=5))
    results = scorer3.get_results()
    for metric in results.keys():
      if metric=='f1_calc':
          result.append(np.average(results[metric]))
      print("%s: %.3f" % (metric, np.average(results[metric])))
parameters= [2,20,100,None]
for K in parameters:
    print("max_leaf_nodes")
    print(K)
    clf = DecisionTreeClassifier(max_leaf_nodes=K,random_state=0)
    cross_val_score(clf, X, y, scoring=scorer3, cv=KFold(n_splits=5))
    results = scorer3.get_results()
    for metric in results.keys():
      print("%s: %.3f" % (metric, np.average(results[metric])))
parameters= [1,20,40,100]
for K in parameters:
    print("min_samples_leaf")
    print(K)
    clf = DecisionTreeClassifier(min_samples_leaf=K,random_state=0)
    cross_val_score(clf, X, y, scoring=scorer3, cv=KFold(n_splits=5))
    results = scorer3.get_results()
    for metric in results.keys():
      print("%s: %.3f" % (metric, np.average(results[metric])))
parameters= [2,5,40,100]
for K in parameters:
    print("min_samples_split")
    print(K)
    clf = DecisionTreeClassifier(min_samples_split=K,random_state=0)
    cross_val_score(clf, X, y, scoring=scorer3, cv=KFold(n_splits=5))
    results = scorer3.get_results()
    for metric in results.keys():
      print("%s: %.3f" % (metric, np.average(results[metric])))
'''------RandomForestClassifier----'''
print("RandomForestClassifier-parameters")
# parameters= [10,50,100,200]
# for K in parameters:
#     print("n_estimators")
#     print(K)
#     clf = RandomForestClassifier(n_estimators=K,random_state=0)
#     cross_val_score(clf, X, y, scoring=scorer3, cv=KFold(n_splits=5))
#     results = scorer3.get_results()
#     for metric in results.keys():
#       print("%s: %.3f" % (metric, np.average(results[metric])))
parameters= [2,10,30]
for K in parameters:
    print("min_samples_split")
    print(K)
    clf = RandomForestClassifier(n_estimators=10,min_samples_split=K,random_state=0)
    cross_val_score(clf, X, y, scoring=scorer3, cv=KFold(n_splits=5))
    results = scorer3.get_results()
    for metric in results.keys():
      print("%s: %.3f" % (metric, np.average(results[metric])))
'''-------ExtraTreesClassifier--------'''
print("ExtraTreesClassifier-parameters")
# parameters= [10,50,100,200]
# for K in parameters:
#     print("n_estimators")
#     print(K)
#     clf = ExtraTreesClassifier(n_estimators=K,random_state=0)
#     cross_val_score(clf, X, y, scoring=scorer3, cv=KFold(n_splits=5))
#     results = scorer3.get_results()
#     for metric in results.keys():
#       print("%s: %.3f" % (metric, np.average(results[metric])))
parameters= [2,10,30]
for K in parameters:
    print("min_samples_split")
    print(K)
    clf = ExtraTreesClassifier(n_estimators=10,min_samples_split=K,random_state=0)
    cross_val_score(clf, X, y, scoring=scorer3, cv=KFold(n_splits=5))
    results = scorer3.get_results()
    for metric in results.keys():
      print("%s: %.3f" % (metric, np.average(results[metric])))
'''----------------------------------------------------------------------------------------------------------------'''
'''---------------------------KNN-TEST3-----------------------------------------------------------------------'''
# print("KnnRegressorTemp")
# parameters= [3,5,15]
# for K in parameters:
#     print("n_estimators")
#     print(K)
#     reg = KNeighborsRegressor(n_neighbors=K,n_jobs=-1,algorithm='kd_tree',leaf_size=500)
#     cross_val_score(reg, X_temp, y, scoring=scorer, cv=KFold(n_splits=3))
#     results = scorer.get_results()
#     for metric in results.keys():
#         print("%s: %.3f" % (metric, np.average(results[metric])))
# print("KnnRegressorAllFeaturs")
# parameters= [3,5,15,121]
# for K in parameters:
#     print("n_estimators")
#     print(K)
#     reg = KNeighborsRegressor(n_neighbors=K,n_jobs=-1,algorithm='kd_tree',leaf_size=500)
#     cross_val_score(reg, X_new, y, scoring=scorer, cv=KFold(n_splits=3))
#     results = scorer.get_results()
#     for metric in results.keys():
#         print("%s: %.3f" % (metric, np.average(results[metric])))
# print("KnnClassifier")
# parameters= [3,5,15,121]
# for K in parameters:
#     print("n_estimators")
#     print(K)
#     reg = KNeighborsClassifier(n_neighbors=K,n_jobs=-1,algorithm='kd_tree',leaf_size=500)
#     cross_val_score(reg, X_new, y, scoring=scorer3, cv=KFold(n_splits=3))
#     results = scorer3.get_results()
#     for metric in results.keys():
#         print("%s: %.3f" % (metric, np.average(results[metric])))
'''-----------------------TEST-4--------------------------------------------'''
Regs={"linera_model" : linear_model.Ridge(random_state=0), "AdaBoostRegressor" : AdaBoostRegressor(random_state=0,n_estimators=10), "GaussianNB" : GaussianNB()}
for reg in Regs :
    print(reg)
    cross_val_score(Regs.get(reg), X_new, y, scoring=scorer, cv=KFold(n_splits=5))
    results = scorer.get_results()
    for metric in results.keys():
      print("%s: %.3f" % (metric, np.average(results[metric])))
'''--------------------------------------------------------------------------'''
'''-------------------------------------------------PART2---TEST1--------------------------------------------'''
'''test  class_whights '''
class_weight1 = {0: 1,
                1: 10,
                2: 100,
                3:1000,
                4:10000,
                5:100000,
                6:1000000
                }
class_weight2 = {0: 0.1,
                1: 10,
                2: 100,
                3:1000,
                4:10000,
                5:100000,
                6:1000000
                }
class_weight3 = {0: 0.01,
                1: 10,
                2: 100,
                3:1000,
                4:10000,
                5:100000,
                6:1000000
                }
class_weight4="balanced"
weights=[class_weight1,class_weight2,class_weight3,class_weight4]
for weight in weights:
    clf=ExtraTreesClassifier(n_estimators=10,random_state=0,class_weight=weight)
    f1=[]
    matthews=[]
    error=[]
    f1_1=[]
    f1_2=[]
    f1_3=[]
    f1_4=[]
    f1_5=[]
    f1_6=[]
    f1_7=[]
    kf = KFold(n_splits=5)
    i=1
    for train_index, test_index in kf.split(df):
        X_train, X_test = X_new.loc[train_index], X_new.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f1_score(y_test,y_pred,average=None))
        f1_1.append(f1_score(y_test,y_pred,average=None)[0])
        f1_2.append(f1_score(y_test,y_pred,average=None)[1])
        f1_3.append(f1_score(y_test,y_pred,average=None)[2])
        f1_4.append(f1_score(y_test, y_pred, average=None)[3])
        f1_5.append(f1_score(y_test, y_pred, average=None)[4])
        f1_6.append(f1_score(y_test, y_pred, average=None)[5])
        if 6 in y_test.values:
            f1_7.append(f1_score(y_test, y_pred, average=None)[6])
        f1.append(f1_score(y_test,y_pred,average='micro'))
        matthews.append(matthews_corrcoef(y_test,y_pred))
        error.append(calc_err(y_test, y_pred))
        if i==1:
            print(confusion_matrix(y_test, y_pred,[0,1,2,3,4,5,6,7,8]))
        i=2
    print('f1 = {}'.format(np.round(mean(f1),4)))
    print('matthews = {}'.format(np.round(mean(matthews), 4)))
    print('error = {}'.format(np.round(mean(error), 4)))
    print('class0 = {}'.format(np.round(mean(f1_1),4)))
    print('class1 = {}'.format(np.round(mean(f1_2),4)))
    print('class2 = {}'.format(np.round(mean(f1_3),4)))
    print('class3 = {}'.format(np.round(mean(f1_4), 4)))
    print('class4 = {}'.format(np.round(mean(f1_5), 4)))
    print('class5 = {}'.format(np.round(mean(f1_6), 4)))
    print('class6 = {}'.format(np.round(mean(f1_7), 4)))
'''-------------------------------------------------------------------------------------------------------------'''
'''-----------------------------------------PART2--TEST2 ---------------------------------------- '''
'''----------------------------under_samples-----------------------------'''
clfs = [ExtraTreesClassifier(random_state=0,n_estimators=10),DecisionTreeClassifier(random_state=0),KNeighborsClassifier(leaf_size=500),GaussianNB()]
for clf in clfs:
    print(clf)
    f1=[]
    matthews=[]
    error=[]
    f1_1=[]
    f1_2=[]
    f1_3=[]
    f1_4=[]
    f1_5=[]
    f1_6=[]
    f1_7=[]
    kf = KFold(n_splits=5)
    i=1
    for train_index, test_index in kf.split(df):
        X_train, X_test = X_new.loc[train_index], X_new.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        rus = RandomUnderSampler(random_state=0)
        X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
        unique, counts = np.unique(y_train_under, return_counts=True)
        if i==1:
            print(dict(zip(unique, counts)))
        clf.fit(X_train_under, y_train_under)
        y_pred = clf.predict(X_test)
        # print(f1_score(y_test,y_pred,average=None))
        f1_1.append(f1_score(y_test,y_pred,average=None)[0])
        f1_2.append(f1_score(y_test,y_pred,average=None)[1])
        f1_3.append(f1_score(y_test,y_pred,average=None)[2])
        f1_4.append(f1_score(y_test, y_pred, average=None)[3])
        f1_5.append(f1_score(y_test, y_pred, average=None)[4])
        f1_6.append(f1_score(y_test, y_pred, average=None)[5])
        if 6 in y_test.values:
            f1_7.append(f1_score(y_test, y_pred, average=None)[6])
        f1.append(f1_score(y_test, y_pred, average='micro'))
        matthews.append(matthews_corrcoef(y_test, y_pred))
        error.append(calc_err(y_test, y_pred))
        if i==1:
            print(confusion_matrix(y_test, y_pred,[0,1,2,3,4,5,6]))
        i=2
    print('f1 = {}'.format(np.round(mean(f1),4)))
    print('class0 = {}'.format(np.round(mean(f1_1),4)))
    print('class1 = {}'.format(np.round(mean(f1_2),4)))
    print('class2 = {}'.format(np.round(mean(f1_3),4)))
    print('class3 = {}'.format(np.round(mean(f1_4), 4)))
    print('class4 = {}'.format(np.round(mean(f1_5), 4)))
    print('class5 = {}'.format(np.round(mean(f1_6), 4)))
    print('class6 = {}'.format(np.round(mean(f1_7), 4)))
'''-----------------end test undersamples----------------------'''
'''-------------------------------------------test divide by 2/3-----------------'''
clfs = [ExtraTreesClassifier(random_state=0,n_estimators=10),KNeighborsClassifier(leaf_size=500),GaussianNB()]
for clf in clfs:
    print(clf)
    f1=[]
    matthews=[]
    error=[]
    f1_1=[]
    f1_2=[]
    f1_3=[]
    f1_4=[]
    f1_5=[]
    f1_6=[]
    f1_7=[]
    kf = KFold(n_splits=5)
    i=1
    for train_index, test_index in kf.split(df):
        X_train, X_test = X_new.loc[train_index], X_new.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        test = pd.concat([X_train,y_train],ignore_index=True,axis=1)
        test.columns = ['Adults','Teen','precentOfAvtala','center','train_co','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','museum_co','police_stations_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP','count_humra']
        # print(test['count_humra'].value_counts())
        test = test.drop(test[test['count_humra'] == 0].sample(frac=.3).index)
        # test = test.drop(test[test['count_humra'] == 1].sample(frac=.75).index)
        X_train=test[['Adults','Teen','precentOfAvtala','center','train_co','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','museum_co','police_stations_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]
        X_train = X_train.iloc[1:]
        y_train = test['count_humra']
        y_train = y_train.iloc[1:]
        unique, counts = np.unique(y_train, return_counts=True)
        if i==1:
            print(dict(zip(unique, counts)))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if i==1:
            print(f1_score(y_test,y_pred,average=None))
        print(f1_score(y_test,y_pred,average=None),  file=open('log.txt', 'w'))
        f1_1.append(f1_score(y_test,y_pred,average=None)[0])
        f1_2.append(f1_score(y_test,y_pred,average=None)[1])
        f1_3.append(f1_score(y_test,y_pred,average=None)[2])
        f1_4.append(f1_score(y_test, y_pred, average=None)[3])
        f1_5.append(f1_score(y_test, y_pred, average=None)[4])
        f1_6.append(f1_score(y_test, y_pred, average=None)[5])
        if 6 in y_test.values:
            f1_7.append(f1_score(y_test, y_pred, average=None)[6])
        f1.append(f1_score(y_test, y_pred, average='micro'))
        matthews.append(matthews_corrcoef(y_test, y_pred))
        error.append(calc_err(y_test, y_pred))
        if i==1:
            print(confusion_matrix(y_test, y_pred,[0,1,2,3,4,5,6]))
        print(confusion_matrix(y_test, y_pred,[0,1,2,3]),  file=open('log2.txt', 'w'))
        i=2
    print("results/n")
    print('f1 = {}'.format(np.round(mean(f1),4)))
    print('class0 = {}'.format(np.round(mean(f1_1),4)))
    print('class1 = {}'.format(np.round(mean(f1_2),4)))
    print('class2 = {}'.format(np.round(mean(f1_3),4)))
    print('class3 = {}'.format(np.round(mean(f1_4), 4)))
    print('class4 = {}'.format(np.round(mean(f1_5), 4)))
    print('class5 = {}'.format(np.round(mean(f1_6), 4)))
    print('class6 = {}'.format(np.round(mean(f1_7), 4)))
'''-----------------------------end test 2/3 class 0----------------------'''
'''----------------------------------------------test  class_whights --------------------------'''
clfs = [ExtraTreesClassifier(random_state=0,n_estimators=10,class_weight="balanced"),DecisionTreeClassifier(random_state=0,class_weight="balanced"),GaussianNB()]
for clf in clfs:
    print(clf)
    f1=[]
    matthews=[]
    error=[]
    f1_1=[]
    f1_2=[]
    f1_3=[]
    f1_4=[]
    f1_5=[]
    f1_6=[]
    f1_7=[]
    kf = KFold(n_splits=5)
    i=1
    for train_index, test_index in kf.split(df):
        X_train, X_test = X_new.loc[train_index], X_new.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f1_score(y_test,y_pred,average=None))
        f1_1.append(f1_score(y_test,y_pred,average=None)[0])
        f1_2.append(f1_score(y_test,y_pred,average=None)[1])
        f1_3.append(f1_score(y_test,y_pred,average=None)[2])
        f1_4.append(f1_score(y_test, y_pred, average=None)[3])
        f1_5.append(f1_score(y_test, y_pred, average=None)[4])
        f1_6.append(f1_score(y_test, y_pred, average=None)[5])
        if 6 in y_test.values:
            f1_7.append(f1_score(y_test, y_pred, average=None)[6])
        f1.append(f1_score(y_test, y_pred, average='micro'))
        matthews.append(matthews_corrcoef(y_test, y_pred))
        error.append(calc_err(y_test, y_pred))
        if i==1:
            print(confusion_matrix(y_test, y_pred,[0,1,2,3,4,5,6]))
        i=2
    print('f1 = {}'.format(np.round(mean(f1),4)))
    print('class0 = {}'.format(np.round(mean(f1_1),4)))
    print('class1 = {}'.format(np.round(mean(f1_2),4)))
    print('class2 = {}'.format(np.round(mean(f1_3),4)))
    print('class3 = {}'.format(np.round(mean(f1_4), 4)))
    print('class4 = {}'.format(np.round(mean(f1_5), 4)))
    print('class5 = {}'.format(np.round(mean(f1_6), 4)))
    print('class6 = {}'.format(np.round(mean(f1_7), 4)))
'''----------------------------end test class_weight----------'''
'''---------------------------------------------------------------------------------------------------------'''
'''------------------------------------TEST 3 Begin---------------------------------------------------------------'''
'''----------------------------------features-Test---------------------------------------------------------'''
'''all features'''
clf = ExtraTreesClassifier(n_estimators=10,random_state=0)
cross_val_score(clf, X, y, scoring=scorer3, cv=KFold(n_splits=5))
results = scorer3.get_results()
for metric in results.keys():
  print("%s: %.3f" % (metric, np.average(results[metric])))
'''without 2 worst'''
# clf = ExtraTreesClassifier(n_estimators=10,random_state=0)
cross_val_score(clf, X_new, y, scoring=scorer3, cv=KFold(n_splits=5))
results = scorer3.get_results()
for metric in results.keys():
  print("%s: %.3f" % (metric, np.average(results[metric])))
X_3_worse=df[['Adults','precentOfAvtala','center','train_co','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','museum_co','police_stations_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]
# clf = ExtraTreesClassifier(n_estimators=10,random_state=0)
cross_val_score(clf, X_3_worse, y, scoring=scorer3, cv=KFold(n_splits=5))
results = scorer3.get_results()
for metric in results.keys():
  print("%s: %.3f" % (metric, np.average(results[metric])))
X_5_worse=df[['Adults','precentOfAvtala','center','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','museum_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]
# clf = ExtraTreesClassifier(n_estimators=10,random_state=0)
cross_val_score(clf, X_5_worse, y, scoring=scorer3, cv=KFold(n_splits=5))
results = scorer3.get_results()
for metric in results.keys():
  print("%s: %.3f" % (metric, np.average(results[metric])))
X_7_worse=df[['Adults','center','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]
# clf = ExtraTreesClassifier(n_estimators=10,random_state=0)
cross_val_score(clf, X_7_worse, y, scoring=scorer3, cv=KFold(n_splits=5))
results = scorer3.get_results()
for metric in results.keys():
  print("%s: %.3f" % (metric, np.average(results[metric])))
plt.show()
'''-------------------------play with classes size with under and over samples technique-----------------------'''
print(clf)
clf = ExtraTreesClassifier(n_estimators=10,random_state=0)
f1=[]
matthews=[]
error=[]
f1_1=[]
f1_2=[]
f1_3=[]
f1_4=[]
f1_5=[]
f1_6=[]
f1_7=[]
kf = KFold(n_splits=5)
i=1
for train_index, test_index in kf.split(df):
    X_train, X_test = X_new.loc[train_index], X_new.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    ros = SMOTE({3 : 7500,4 : 2000, 5: 500, 6:200},random_state=0)
    X_train_under, y_train_under = ros.fit_resample(X_train, y_train)
    clf.fit(X_train_under, y_train_under)
    y_pred = clf.predict(X_test)
    print(f1_score(y_test,y_pred,average=None))
    f1_1.append(f1_score(y_test,y_pred,average=None)[0])
    f1_2.append(f1_score(y_test,y_pred,average=None)[1])
    f1_3.append(f1_score(y_test,y_pred,average=None)[2])
    f1_4.append(f1_score(y_test, y_pred, average=None)[3])
    f1_5.append(f1_score(y_test, y_pred, average=None)[4])
    f1_6.append(f1_score(y_test, y_pred, average=None)[5])
    if 6 in y_test.values:
        f1_7.append(f1_score(y_test, y_pred, average=None)[6])
    f1.append(f1_score(y_test, y_pred, average='micro'))
    matthews.append(matthews_corrcoef(y_test, y_pred))
    error.append(calc_err(y_test, y_pred))
    if i==1:
        print(confusion_matrix(y_test, y_pred,[0,1,2,3,4,5,6]))
    i=2
print('f1 = {}'.format(np.round(mean(f1),4)))
print('class0 = {}'.format(np.round(mean(f1_1),4)))
print('class1 = {}'.format(np.round(mean(f1_2),4)))
print('class2 = {}'.format(np.round(mean(f1_3),4)))
print('class3 = {}'.format(np.round(mean(f1_4), 4)))
print('class4 = {}'.format(np.round(mean(f1_5), 4)))
print('class5 = {}'.format(np.round(mean(f1_6), 4)))
print('class6 = {}'.format(np.round(mean(f1_7), 4)))
clf = ExtraTreesClassifier(n_estimators=10,random_state=0)
f1=[]
matthews=[]
error=[]
f1_1=[]
f1_2=[]
f1_3=[]
f1_4=[]
f1_5=[]
f1_6=[]
f1_7=[]
kf = KFold(n_splits=5)
i=1
for train_index, test_index in kf.split(df):
    X_train, X_test = X_new.loc[train_index], X_new.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    rus = NearMiss({0 : 150000,1 : 80000, 2: 20000},random_state=0)
    X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
    ros = SMOTE({3 : 7500,4 : 2000, 5: 500, 6:200},random_state=0)
    X_train_under, y_train_under = ros.fit_resample(X_train_under, y_train_under)
    clf.fit(X_train_under, y_train_under)
    y_pred = clf.predict(X_test)
    print(f1_score(y_test,y_pred,average=None))
    f1_1.append(f1_score(y_test,y_pred,average=None)[0])
    f1_2.append(f1_score(y_test,y_pred,average=None)[1])
    f1_3.append(f1_score(y_test,y_pred,average=None)[2])
    f1_4.append(f1_score(y_test, y_pred, average=None)[3])
    f1_5.append(f1_score(y_test, y_pred, average=None)[4])
    f1_6.append(f1_score(y_test, y_pred, average=None)[5])
    if 6 in y_test.values:
        f1_7.append(f1_score(y_test, y_pred, average=None)[6])
    f1.append(f1_score(y_test, y_pred, average='micro'))
    matthews.append(matthews_corrcoef(y_test, y_pred))
    error.append(calc_err(y_test, y_pred))
    if i==1:
        print(confusion_matrix(y_test, y_pred,[0,1,2,3,4,5,6]))
    i=2
print('f1 = {}'.format(np.round(mean(f1),4)))
print('class0 = {}'.format(np.round(mean(f1_1),4)))
print('class1 = {}'.format(np.round(mean(f1_2),4)))
print('class2 = {}'.format(np.round(mean(f1_3),4)))
print('class3 = {}'.format(np.round(mean(f1_4), 4)))
print('class4 = {}'.format(np.round(mean(f1_5), 4)))
print('class5 = {}'.format(np.round(mean(f1_6), 4)))
print('class6 = {}'.format(np.round(mean(f1_7), 4)))
# clf = ExtraTreesClassifier(n_estimators=10,random_state=0)
# f1=[]
# matthews=[]
# error=[]
# f1_1=[]
# f1_2=[]
# f1_3=[]
# f1_4=[]
# f1_5=[]
# f1_6=[]
# f1_7=[]
# kf = KFold(n_splits=5)
# i=1
# for train_index, test_index in kf.split(df):
#     X_train, X_test = X_new.loc[train_index], X_new.loc[test_index]
#     y_train, y_test = y.loc[train_index], y.loc[test_index]
#     print(y_train.value_counts())
#     rus = TomekLinks({0 : 150000,1 : 80000, 2: 20000},random_state=0)
#     X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
#     ros = SMOTE({3 : 7500,4 : 2000, 5: 500, 6:200},random_state=0)
#     X_train_under, y_train_under = ros.fit_resample(X_train_under, y_train_under)
#     unique, counts = np.unique(y_train_under, return_counts=True)
#     if i==1:
#         print(dict(zip(unique, counts)))
#     clf.fit(X_train_under, y_train_under)
#     y_pred = clf.predict(X_test)
#     print(f1_score(y_test,y_pred,average=None))
#     f1_1.append(f1_score(y_test,y_pred,average=None)[0])
#     f1_2.append(f1_score(y_test,y_pred,average=None)[1])
#     f1_3.append(f1_score(y_test,y_pred,average=None)[2])
#     f1_4.append(f1_score(y_test, y_pred, average=None)[3])
#     f1_5.append(f1_score(y_test, y_pred, average=None)[4])
#     f1_6.append(f1_score(y_test, y_pred, average=None)[5])
#     if 6 in y_test.values:
#         f1_7.append(f1_score(y_test, y_pred, average=None)[6])
#     f1.append(f1_score(y_test, y_pred, average='micro'))
#     matthews.append(matthews_corrcoef(y_test, y_pred))
#     error.append(calc_err(y_test, y_pred))
#     if i==1:
#         print(confusion_matrix(y_test, y_pred,[0,1,2,3,4,5,6]))
#     i=2
# print('f1 = {}'.format(np.round(mean(f1),4)))
# print('class0 = {}'.format(np.round(mean(f1_1),4)))
# print('class1 = {}'.format(np.round(mean(f1_2),4)))
# print('class2 = {}'.format(np.round(mean(f1_3),4)))
# print('class3 = {}'.format(np.round(mean(f1_4), 4)))
# print('class4 = {}'.format(np.round(mean(f1_5), 4)))
# print('class5 = {}'.format(np.round(mean(f1_6), 4)))
# print('class6 = {}'.format(np.round(mean(f1_7), 4)))
# '''-----------------------------------------'''
# '''if you want to save the model'''
# pickle.dump(model, open('modelknn20'.sav', 'wb'))
# loaded_model = pickle.load(open("2modelknn20.sav", 'rb'))
# pred=loaded_model.predict(X_test) #make prediction on test set
# print("MSE:")
# print(mean_squared_error(y_test,pred))#calculate rmse
# print("R2_score")
# print(r2_score(y_test, pred))
'''------------------------------------'''
'''last test'''
X_7_worse=df[['Adults','center','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]
clf = ExtraTreesClassifier(n_estimators=10,random_state=0)
cross_val_score(clf, X_7_worse, y, scoring=scorer3, cv=KFold(n_splits=5))
results = scorer3.get_results()
for metric in results.keys():
  print("%s: %.3f" % (metric, np.average(results[metric])))
X_7_worse=df[['Adults','center','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]
clf = ExtraTreesClassifier(n_estimators=10,random_state=0,class_weight="balanced")
cross_val_score(clf, X_7_worse, y, scoring=scorer3, cv=KFold(n_splits=5))
results = scorer3.get_results()
for metric in results.keys():
  print("%s: %.3f" % (metric, np.average(results[metric])))
X_7_worse=df[['Adults','center','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]
clf = ExtraTreesClassifier(n_estimators=200,random_state=0,class_weight="balanced")
cross_val_score(clf, X_7_worse, y, scoring=scorer3, cv=KFold(n_splits=5))
results = scorer3.get_results()
for metric in results.keys():
  print("%s: %.3f" % (metric, np.average(results[metric])))
'''-----------------------------------------------RANDOM vs CLASSIFIER TEST------------------------------------'''
df = pd.read_csv("data.csv")
df.fillna(0 , inplace=True)
X_new_random=df[['Adults','Teen','precentOfAvtala','center','train_co','parking','dogs_coordinates','resturants_onlyCoordinats','shopcenter_co','speedCamera_co','industrialarea_co','museum_co','police_stations_co','barandpub','education_co','gas_station_only_israel','sug_yom','ped','rain','speed','MAX_TEMP','MIN_TEMP']]
y=df['count_humra']
train_pct_index = int(0.9 * len(X_new_random))
X_train_All_Years, X_test_All_Years = X_new_random[:train_pct_index], X_new_random[train_pct_index:]
y_train_All_Years, y_test_All_Years = y[:train_pct_index], y[train_pct_index:]
clf = ExtraTreesClassifier(n_estimators=15,random_state=0,class_weight="balanced")
clf.fit(X_train_All_Years,y_train_All_Years)
y_pred=clf.predict(X_test_All_Years)
y_random= [random.randint(0,5) for i in range(y_pred.size)]
# print(roc_auc_score(y_test, y_pred))
print(matthews_corrcoef(y_test_All_Years,y_pred))
print(f1_calc(y_test_All_Years,y_pred))
print(confusion_matrix(y_test_All_Years,y_pred))
print(matthews_corrcoef(y_test_All_Years,y_random))
print(f1_calc(y_test_All_Years,y_random))
print(confusion_matrix(y_test_All_Years,y_random))

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plot_confusion_matrix(y_test_All_Years, y_pred, classes=[0,1,2,3,4,5], normalize=True,
                      title='Normalized confusion matrix')
plot_confusion_matrix(y_test_All_Years, y_random, classes=[0,1,2,3,4,5], normalize=True,
                      title='Normalized confusion matrix')
'''----------------------------------------------------------------------------------------'''
'''-----------------------------plot with None-------------------------------------'''
# result=np.array([0.885,0.897,0.901,0.903])
# parameters= np.array([2,5,200,'None'])
# plt.xticks(range(len(result)), parameters)
# plt.xlabel("max_depth")
# plt.ylabel("f1_score")
# plt.title("effect of depth on model performence")
# plt.plot(result)
# plt.show()
# plt.close()
# result=np.array([0.917,0.926,0.929,0.931,0.932])
# results2=np.array([0.521,0.596,0.62,0.631,0.638])
# parameters= np.array([0,4,7,10,14])
# plt.xticks(range(len(result)), parameters)
# plt.xlabel("K_Neighbors")
# plt.ylabel("F1-score")
# plt.title("effect of K_Neighbors on model performence")
# plt.plot(result,marker='o',color='red',label="f1-score", linewidth=2)
# plt.plot(results2,marker='o',color='olive',label="MCC")
# plt.legend()
# plt.show()
'''-----------------------------------------------------------------------------------------------------------'''
plt.show()