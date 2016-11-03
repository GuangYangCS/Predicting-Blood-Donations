import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import urllib2
from sklearn import metrics
from sklearn import preprocessing
import ssl

context = ssl._create_unverified_context()
fold = 30

###get data from url###
urlForTransfusion = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
dataForTransfusion = urllib2.urlopen(urlForTransfusion, context=context)
datasetForTransfusion = np.loadtxt(dataForTransfusion, delimiter=",", skiprows=1)
#print datasetForTransfusion.shape
#print datasetForTransfusion[0]

###get x and y from dataset###
x_Transfusion = datasetForTransfusion[:, 0:4]
x_Normalized = preprocessing.normalize(x_Transfusion, norm='l2')
y_Transfusion = datasetForTransfusion[:, 4]


###SVM###
print 'SVM'
svc = svm.SVC(kernel='rbf')
for i in range(10, 100):
    scoresForSVM = cross_val_score(svc, x_Transfusion, y_Transfusion, cv=i)
    #print scoresForSVM
    #print 'The average accuracy of SVM for blood-transfusion is', scoresForSVM.mean()
    print scoresForSVM.mean()

'''
svc.fit(x_Normalized, y_Transfusion)
expected = y_Transfusion
predicted = svc.predict(x_Normalized)
print 'The classification report of SVM for Blood-transfusion:\n', metrics.classification_report(expected, predicted)
#print 'The confusion matrix of SVM for Blood-transfusion:\n',metrics.confusion_matrix(expected, predicted)
'''


###bagging###
print '\nBagging'

bagging = BaggingClassifier(base_estimator=None,max_samples=0.5, max_features=0.5)
for i in range(10, 100):
    scoresForBagging = cross_val_score(bagging, x_Normalized, y_Transfusion, cv=i)
    #print scoresForBagging
    # #print 'The average accuracy of Bagging for blood-transfusion is', scoresForBagging.mean()
    print scoresForBagging.mean()

'''
bagging.fit(x_Normalized, y_Transfusion)
expected = y_Transfusion
predicted = bagging.predict(x_Normalized)
print 'The classification report of Bagging for Blood-transfusion:\n', metrics.classification_report(expected, predicted)
#print 'The confusion matrix of Bagging for Blood-transfusion:\n',metrics.confusion_matrix(expected, predicted)
'''


###Random Forests###
print '\nRandom Forests'

for i in range(10, 100):
    randomForest = RandomForestClassifier(n_estimators=10, min_samples_split=5, random_state=0, max_depth=None, max_features=0.5)
    scoresForRandomForests = cross_val_score(randomForest, x_Normalized, y_Transfusion, cv=i)
    # print scoresForRandomForests
    #print 'For the estimator', i, ', the average accuracy of Random Forests for blood-transfusion is', scoresForRandomForests.mean()
    print scoresForRandomForests.mean()
    #print scoresForRandomForests

'''
randomForest = RandomForestClassifier(n_estimators=10, min_samples_split=5, random_state=0, max_depth=None, max_features=0.5)
randomForest.fit(x_Normalized, y_Transfusion)
expected = y_Transfusion
predicted = randomForest.predict(x_Normalized)
print 'The classification report of Random Forest for Blood-transfusion:\n', metrics.classification_report(expected, predicted)
print 'The confusion matrix of Random Forest for Blood-transfusion:\n',metrics.confusion_matrix(expected, predicted)
'''


###AdaBoost###
print '\nAdaBoost'
adaBoost = AdaBoostClassifier(n_estimators=500,  learning_rate=0.01)
for i in range(10, 100):
    scoresForAdaBoost = cross_val_score(adaBoost, x_Normalized, y_Transfusion, cv=i)
    #print scoresForAdaBoost
    #print 'The average accuracy of AdaBoost for blood-transfusion is', scoresForAdaBoost.mean()
    print scoresForAdaBoost.mean()
    #print scoresForAdaBoost

'''
adaBoost.fit(x_Normalized, y_Transfusion)
expected = y_Transfusion
predicted = adaBoost.predict(x_Normalized)
print 'The classification report of Adaboost for Blood-transfusion:\n', metrics.classification_report(expected, predicted)
#print 'The confusion matrix of Adaboost for Blood-transfusion:\n',metrics.confusion_matrix(expected, predicted)
'''


###Gradient Boosting###
print '\nGradient Boosting:'

for i in range(10, 100):
    gradientBoost = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=2, random_state=0)
    scoresForGradientBoost = cross_val_score(gradientBoost, x_Normalized, y_Transfusion, cv=i)
    # print scoresForAdaBoost
    print scoresForGradientBoost.mean()
    #print 'For the fold',i ,', the average accuracy of Gradient Boosting for blood-transfusion is', scoresForGradientBoost.mean()

'''
gradientBoost.fit(x_Normalized, y_Transfusion)
expected = y_Transfusion
predicted = gradientBoost.predict(x_Normalized)
print 'The classification report of gradientBoost for Blood-transfusion:\n', metrics.classification_report(expected, predicted)
#print 'The confusion matrix of gradientBoost for Blood-transfusion:\n',metrics.confusion_matrix(expected, predicted)
'''