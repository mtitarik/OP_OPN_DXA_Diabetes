#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
#from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score





#Evaluation Metrics

def compute_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    mcc = (tp*tn - fp*fn) / (np.sqrt(  (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)  ) + 1e-8)
    return round(mcc,3)
    
def compute_sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp/(tp+fn)
    return round(sensitivity,3)
    
def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn/(fp+tn)
    return round(specificity,3)
    
def compute_accuracy(y_true, y_pred):
    accuracy = (y_true==y_pred).sum()/len(y_true)
    return round(accuracy,3)

# def compute_precision(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = cm.ravel()
#     precision = tn/(tn+fp)
#     return round(precision,3)
    
def compute_precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    if( tp+fp )==0 :
        precision = 0;
    else:    
        precision = tp/(tp+fp)
    return round(precision,3)    

def compute_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro" , zero_division=1)
    return round(f1,3)




#
# ####################################
# 
# Manual calculations of performance measures 
#
import numpy as np
import pandas as pd
from sklearn.metrics import  precision_recall_curve, auc
from catboost import CatBoostClassifier

from xgboost import XGBClassifier
# import xgboost as xgb

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold 
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from pandas import MultiIndex, Int64Index
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score

classifiers = [
              
               AdaBoostClassifier(n_estimators=100, random_state=42),
               XGBClassifier(eval_metric='rmse', use_label_encoder=False, verbosity = 0, random_state=42),
               CatBoostClassifier(verbose = False, random_state=42),
               GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
               RandomForestClassifier(random_state=42), 
               tree.DecisionTreeClassifier(random_state=42),
               LinearDiscriminantAnalysis(),
               #LogisticRegression(random_state=42),
               BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=42),
               svm.SVC(random_state=42, probability=True),
               GaussianNB()
              
              ]

import matplotlib.pyplot as plt

with open('./Result_10Fold_CrossValidationRuns_NoBMD.txt', 'w') as cvresults:
    
    cvresults.write("Classifer,"+"Sen(%)," + "Spe(%)," + "ACC(%)," + "Pre(%)," + "F1(%)," + "MCC," + "\n")

    df = pd.read_csv("./FeatureVector_Freezed_Final_NoBMD.csv")
    
    
    print(df['class'].value_counts()) 
    y = df["class"]
    X = df.drop(["class"], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    k = 10
    #kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for clf in classifiers:
        model = clf
        mcc_score = []
        sen_score = []
        spe_score = []
        acc_score = []
        pre_score = []
        F1_score  = []

        # sorted(sklearn.metrics.SCORERS.keys())
        scoring_functions = {'balanced_acc': 'balanced_accuracy',
                             'acc': make_scorer(accuracy_score),
                            'precision': 'precision',
                            'recall': 'recall',
                            'sensitivity': make_scorer(compute_sensitivity),
                            'specificity': make_scorer(compute_specificity),
                            'f1': make_scorer(compute_f1),
                            'mcc': make_scorer(compute_mcc),
                            'neg_log_loss': 'neg_log_loss',
                            'neg_mean_squared_error': 'neg_mean_squared_error'
                            }
        scores = cross_validate(clf, X, y, cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=42), scoring=scoring_functions)
        #print(scores)
        avg_mean_squared_error = np.nanmean(-1*scores['test_neg_mean_squared_error'])
        #neg_mean_squared_error = neg_mean_squared_errors[-1]
        
        avg_log_loss = np.nanmean(-1*scores['test_neg_log_loss'])
        #neg_log_loss = neg_log_losses[-1]
        
        avg_mcc_score = np.nanmean(scores['test_mcc'])
        avg_sen_score = np.nanmean(scores['test_sensitivity'])
        avg_spe_score = np.nanmean(scores['test_specificity'])
        avg_acc_score = np.nanmean(scores['test_acc'])
        avg_f1_score = np.nanmean(scores['test_f1'])
        avg_pre_score = np.nanmean(scores['test_precision'])
        
        #plt.plot(-1*neg_mean_squared_errors)
        #plt.title(clf.__class__.__name__+': mean_squared_error ('+str(-1*neg_mean_squared_error)+')')
        #plt.show()
        
        #plt.plot(-1*neg_log_losses)
        #plt.title(clf.__class__.__name__+': log_loss ('+str(-1*neg_log_loss)+')')
        #plt.show()
        
    

        print(clf)
        #print('Sensitivity of each fold - {}'.format(sen_score))
        print('Avg Sensitivity : {}'.format(round(avg_sen_score,3)*100))

        #print('Specifity of each fold - {}'.format(spe_score))
        print('Avg Specifity : {}'.format(round(avg_spe_score,3)*100))

        #print('accuracy of each fold - {}'.format(acc_score))
        print('Avg accuracy : {}'.format(round(avg_acc_score,3)*100))

        #print('Precision of each fold - {}'.format(pre_score))
        print('Avg Precision : {}'.format(round(avg_pre_score,3)*100))


        #print('F1 of each fold - {}'.format(F1_score))
        print('Avg F1 : {}'.format(round(avg_f1_score,3)*100))

        #print('mcc of each fold - {}'.format(mcc_score))
        print('Avg mcc : {}'.format(round(avg_mcc_score,3)*100))
        
        print('Avg Log Loss : {}'.format(round(avg_log_loss,3)*100))

        print('Avg MSE : {}'.format(round(avg_mean_squared_error,3)*100))

        #print('AUC of each fold - {}'.format(Roc_AUC_score))
        #print('Avg AUC : {}'.format(round(avg_AUC_score,3)*100))
        #print('Avg PR AUC : {}'.format(round(avg_PRAUC_score,3)*100))

        ResultsSummary = (str(round(avg_sen_score,3)*100) + "," + 
                          str(round(avg_spe_score,3)*100) + "," + 
                          str(round(avg_acc_score,3)*100) + "," + 
                          str(round(avg_pre_score,3)*100) + "," +
                          str(round(avg_f1_score,3)*100) + "," +
                          str(round(avg_mcc_score,3 )*100) 
                          #str(round(avg_AUC_score,3)*100) + "," +
                          #str(round(avg_PRAUC_score,3)*100)
                         )
        cvresults.write(str(clf)+","+str(ResultsSummary + "\n"))
        print("========================================================")

