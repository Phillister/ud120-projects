#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

clf = svm.SVC(C=10000.0, kernel="rbf")

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]


t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
print clf
predict = clf.predict(features_test)
print accuracy_score(predict, labels_test)
print "prediction time:", round(time()-t1, 3), "s"

print "Result for 10:", predict[10]
print "Result for 26:", predict[26]
print "Result for 50:", predict[50]

counter = 0
i = 0
while i<len(predict):
    if predict[i] == 1:
        counter += 1
    i += 1
    
print "Amount of E-Mails by Chris:", counter
#########################################################


