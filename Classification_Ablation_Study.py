#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.metrics import  precision_recall_curve, auc
from catboost import CatBoostClassifier
#from xgboost import XGBClassifier
import xgboost as xgb
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


#Evaluation Metrics

def compute_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix", cm)
    tn, fp, fn, tp = cm.ravel()
    print( "tn=",tn, "fp=",fp, "fn=",fn, "tp=",tp)
    mcc = (tp*tn - fp*fn) / (np.sqrt(  (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)  ) + 1e-8)
    print("mcc = ", mcc)
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

#def compute_precision(y_true, y_pred):
#    cm = confusion_matrix(y_true, y_pred)
#    tn, fp, fn, tp = cm.ravel()
#    precision = tn/(tn+fp)
#    return round(precision,3)
    
def compute_precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
#     print(" tp " + str(tp)  + " fn " +  str(fn) + " fp " +  str(fp) + " tn " + str(tn) ) 
    if( tp+fp )==0 :
        precision = 0;
    else:    
        precision = tp/(tp+fp)
    return round(precision,3)    

def compute_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro" , zero_division=1)
    return round(f1,3)





# read feature file

df_master = pd.read_csv("FeatureVector_Freezed_Final_NoBMD.csv")
print(df_master.shape)
featurelist = list(df_master.columns)
print(len(featurelist))
print(featurelist)
#print(df['class'].value_counts()) 


# In[51]:



BONE_MASS = ['CT_BONE_MASS_ANDROID VISCERAL (CoreScan)', 'CT_BONE_MASS_ANDROID', 'CT_BONE_MASS_ARMS', 'CT_BONE_MASS_GYNOID', 'CT_BONE_MASS_LEGS', 'CT_BONE_MASS_TOTAL', 'CT_BONE_MASS_TRUNK', 'class']

FAT_MASS = ['CT_FAT_MASS_ANDROID VISCERAL (CoreScan)', 'CT_FAT_MASS_ANDROID', 'CT_FAT_MASS_Appendicular Lean Mass/Height_ for Age Index', 'CT_FAT_MASS_ARMS', 'CT_FAT_MASS_GYNOID', 
            'CT_FAT_MASS_Legs/Total Fat Mass Ratio for Age Index', 'CT_FAT_MASS_LEGS', 'CT_FAT_MASS_Limbs/Trunk Fat Mass Ratio for Age Index', 'CT_FAT_MASS_Total Body Fat Mass/Height_ for Age Index', 
            'CT_FAT_MASS_Total Body Lean Mass/Height_ for Age Index', 'CT_FAT_MASS_TOTAL', 'CT_FAT_MASS_Trunk/Legs %Fat Ratio for Age Index', 'CT_FAT_MASS_Trunk/Limbs Fat Mass Ratio for Age Index', 
            'CT_FAT_MASS_Trunk/Total Fat Mass Ratio for Age Index', 'CT_FAT_MASS_TRUNK', 'class']
            
            
#             'CT_FAT_MASS_ANDROID VISCERAL (CoreScan)', 'CT_FAT_MASS_ANDROID', 'CT_FAT_MASS_Appendicular Lean Mass/Height_ for Age Index', 
#             'CT_FAT_MASS_ARMS', 'CT_FAT_MASS_GYNOID', 'CT_FAT_MASS_Legs/Total Fat Mass Ratio for Age Index', 'CT_FAT_MASS_LEGS', 'CT_FAT_MASS_Limbs/Trunk Fat Mass Ratio for Age Index', 
#             'CT_FAT_MASS_Total Body Fat Mass/Height_ for Age Index', 'CT_FAT_MASS_Total Body Lean Mass/Height_ for Age Index', 'CT_FAT_MASS_TOTAL', 'CT_FAT_MASS_Trunk/Legs %Fat Ratio for Age Index', 
#             'CT_FAT_MASS_Trunk/Limbs Fat Mass Ratio for Age Index', 'CT_FAT_MASS_Trunk/Total Fat Mass Ratio for Age Index', 'CT_FAT_MASS_TRUNK', 'class']


LEAN_MASS = ['CT_LEAN_MASS_ANDROID VISCERAL (CoreScan)', 'CT_LEAN_MASS_ANDROID', 'CT_LEAN_MASS_ARMS', 'CT_LEAN_MASS_GYNOID', 'CT_LEAN_MASS_LEGS', 'CT_LEAN_MASS_TOTAL', 'CT_LEAN_MASS_TRUNK', 'class']

BMC = ['DT_BMC_ARMS', 'DT_BMC_HEAD', 'DT_BMC_L1', 'DT_BMC_L1_L2', 'DT_BMC_L1_L3', 'DT_BMC_L1_L4', 'DT_BMC_L2', 'DT_BMC_L2_L3', 'DT_BMC_L2_L4', 'DT_BMC_L3', 'DT_BMC_L3_L4', 'DT_BMC_L4', 
       'DT_BMC_LEGS', 'DT_BMC_LOWER NECK', 'DT_BMC_NECK ', 'DT_BMC_PELVIS', 'DT_BMC_RIBS', 'DT_BMC_SHAFT', 'DT_BMC_SPINE', 'DT_BMC_TOTAL', 'DT_BMC_TROCH', 'DT_BMC_TRUNK', 
       'DT_BMC_UPPER NECK', 'DT_BMC_WARDS', 'class']

ANTHRO = ['DT_AREA_ARMS', 'DT_AREA_HEAD', 'DT_AREA_L1', 'DT_AREA_L1_L2', 'DT_AREA_L1_L3', 'DT_AREA_L1_L4', 'DT_AREA_L2', 'DT_AREA_L2_L3', 'DT_AREA_L2_L4', 'DT_AREA_L3', 'DT_AREA_L3_L4', 'DT_AREA_L4', 
          'DT_AREA_LEGS', 'DT_AREA_LOWER NECK', 'DT_AREA_NECK ', 'DT_AREA_PELVIS', 'DT_AREA_RIBS', 'DT_AREA_SHAFT', 'DT_AREA_SPINE', 'DT_AREA_TOTAL', 'DT_AREA_TROCH', 'DT_AREA_TRUNK', 'DT_AREA_UPPER NECK', 
          'DT_AREA_WARDS', 'DT_AVG_HEIGHT_L1', 'DT_AVG_HEIGHT_L1_L2', 'DT_AVG_HEIGHT_L1_L3', 'DT_AVG_HEIGHT_L1_L4', 'DT_AVG_HEIGHT_L2', 'DT_AVG_HEIGHT_L2_L3', 'DT_AVG_HEIGHT_L2_L4', 'DT_AVG_HEIGHT_L3', 
          'DT_AVG_HEIGHT_L3_L4', 'DT_AVG_HEIGHT_L4', 'DT_AVG_WIDTH_L1', 'DT_AVG_WIDTH_L1_L2', 'DT_AVG_WIDTH_L1_L3', 'DT_AVG_WIDTH_L1_L4', 'DT_AVG_WIDTH_L2', 'DT_AVG_WIDTH_L2_L3', 'DT_AVG_WIDTH_L2_L4', 
          'DT_AVG_WIDTH_L3', 'DT_AVG_WIDTH_L3_L4', 'DT_AVG_WIDTH_L4', 'MT_AVG_HEIGHT_HIP AXIS LENGTH', 'class']

# BMD = ['DT_BMD_ARMS', 'DT_BMD_HEAD', 'DT_BMD_L1', 'DT_BMD_L1_L2', 'DT_BMD_L1_L3', 'DT_BMD_L1_L4', 'DT_BMD_L2', 'DT_BMD_L2_L3', 'DT_BMD_L2_L4', 'DT_BMD_L3', 'DT_BMD_L3_L4', 'DT_BMD_L4', 
#        'DT_BMD_LEGS', 'DT_BMD_LOWER NECK', 'DT_BMD_NECK ', 'DT_BMD_PELVIS', 'DT_BMD_RIBS', 'DT_BMD_SHAFT', 'DT_BMD_SPINE', 'DT_BMD_TOTAL', 'DT_BMD_TROCH', 'DT_BMD_TRUNK', 'DT_BMD_UPPER NECK', 
#        'DT_BMD_WARDS']


print(len(BONE_MASS))
print(len(FAT_MASS))
print(len(LEAN_MASS))
print(len(BMC))
print(len(ANTHRO))
#print(len(BMD))




df_BONE_MASS = df_master[BONE_MASS]
print(df_BONE_MASS.shape)
#display(df_BONE_MASS)

df_FAT_MASS = df_master[FAT_MASS]
print(df_FAT_MASS.shape)
#display(df_FAT_MASS)

df_LEAN_MASS = df_master[LEAN_MASS]
print(df_LEAN_MASS.shape)
#display(df_LEAN_MASS)

df_BMC = df_master[BMC]
print(df_BMC.shape)
#display(df_BMC)

df_ANTHRO = df_master[ANTHRO]
print(df_ANTHRO.shape)
#display(df_ANTHRO)


# In[54]:


#
# ####################################
# 
# Manual calculations of performance measures 
#

#dfs = [df_BONE_MASS, df_FAT_MASS, df_LEAN_MASS, df_BMC, df_ANTHRO ]
#dfs = [df_LEAN_MASS, df_FAT_MASS, df_BONE_MASS, df_BMC, df_ANTHRO]
#data_set = ['LEAN_MASS', 'FAT_MASS','BONE_MASS','BMC','ANTHRO']

dfs = [df_FAT_MASS]
data_set = ['FAT_MASS']



classifiers = [AdaBoostClassifier(n_estimators=100, random_state=42),
               xgb.XGBClassifier(eval_metric='rmse', use_label_encoder=False, silent=True, verbosity=0,  random_state=42),
               CatBoostClassifier(verbose = False, random_state=42),
               GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
               RandomForestClassifier(random_state=42),
               tree.DecisionTreeClassifier(random_state=42),
               LinearDiscriminantAnalysis(),
               BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=42),
               svm.SVC(random_state=42),
               GaussianNB() 
              ]




with open('./Ablation_Study_DXA_NoBMD_10Fold_CrossValidationRuns.txt', 'w') as cvresults:
    ds_index = 0
    for df in dfs:
        print("ds index = ",ds_index)
        cvresults.write(str(data_set[ds_index]) + "\n")
        cvresults.write("Classifer,"+"Sen(%)," + "Spe(%)," + "ACC(%)," + "Pre(%)," + "F1(%)," + "MCC," + "\n")
        print(df['class'].value_counts()) 
        y = df["class"]
        X = df.drop(["class"], axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X)
        k = 10
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        for clf in classifiers:
            model = clf
            mcc_score = []
            sen_score = []
            spe_score = []
            acc_score = []
            pre_score = []
            F1_score  = []

            for train_index , test_index in kf.split(X,y):
                X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
                model.fit(X_train,y_train)
                pred_values = model.predict(X_test)
                #print(classification_report(y_test, pred_values))
                mcc = compute_mcc(y_test, pred_values)
                sen = compute_sensitivity(y_test, pred_values)
                spe = compute_specificity(y_test, pred_values)
                acc = compute_accuracy(y_test, pred_values)
                pre = compute_precision(y_test, pred_values)
                F1  = compute_f1(y_test, pred_values)
                #F1 = (2*pre*sen)/(pre+sen)
                # precision, recall, thresholds = precision_recall_curve(y_test, pred_values)
                # auc_precision_recall = auc(recall, precision)
                mcc_score.append(mcc)
                sen_score.append(sen)
                spe_score.append(spe)
                pre_score.append(pre)
                acc_score.append(acc)
                F1_score.append(F1)

            avg_mcc_score = sum(mcc_score) / k
            avg_sen_score = sum(sen_score) / k
            avg_spe_score = sum(spe_score) / k
            avg_acc_score = sum(acc_score) / k
            avg_pre_score = sum(pre_score) / k
            avg_f1_score = sum(F1_score)  / k

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

            #print('AUC of each fold - {}'.format(Roc_AUC_score))
            #print('Avg AUC : {}'.format(round(avg_AUC_score,3)*100))
            #print('Avg PR AUC : {}'.format(round(avg_PRAUC_score,3)*100))

            ResultsSummary = (str(round(avg_sen_score,3)*100) + "," + 
                              str(round(avg_spe_score,3)*100) + "," + 
                              str(round(avg_acc_score,3)*100) + "," + 
                              str(round(avg_pre_score,3)*100) + "," +
                              str(round(avg_f1_score,3)*100) + "," +
                              str(round(avg_mcc_score,3 )*100) 
                             )
            cvresults.write(str(clf)+","+str(ResultsSummary + "\n"))
        ds_index+=1
        print("========================================================")


