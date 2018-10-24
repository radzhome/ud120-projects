#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

    http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# train
# create train set (fit) based on what we know is for sure,
# feature train set and the target labels
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# predict
# create prediction based on feature test data
t0 = time()
pr = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

# accuracy
# label test has the correct labels for each point
# check how well they math up with the predicted data
from sklearn.metrics import accuracy_score
score = accuracy_score(labels_test, pr)
print(score)
#########################################################


