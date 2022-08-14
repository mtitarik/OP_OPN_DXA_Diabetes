#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import xgboost as xgb 
import catboost  

from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler  

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

import time
import random

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt import hp



import sklearn
sklearn.__version__


# #### Set seeds



GLOBAL_RANDOM_STATE = 86

# Function for setting the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# #### For pretty printing


def setup_pretty_printing():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)


# #### Data Loaders (Synthetic and Real)



class DataLoader():
    
   
    @staticmethod
    def load_real_dataset(datasetfilepath, cols_to_drop, target_col):
        df = pd.read_csv(datasetfilepath)
        
        df.drop(cols_to_drop, axis = 1, inplace=True)
        
        X = df.drop(target_col, axis = 1)
        y = df[target_col]
        feature_names = X.columns
        
        return X.values, y.values, feature_names
        


# #### Metrics

# In[6]:


class Metrics():

    @staticmethod
    def sensitivity_metric(preds_prob, dtrain): # builtin: recall_macro???
        y = dtrain.get_label()
        y_preds = preds_prob >= 0.5
        
        conf_mat = confusion_matrix(y, y_preds)
        tn, fp, fn, tp = conf_mat.ravel()
        sensitivity = tp/(tp + fn)
            
        balanced_acc = balanced_accuracy_score(y, y_preds)
        #print('balanced_acc',balanced_acc)
        return 'sensitivity', sensitivity
    
    @staticmethod
    def compute_mcc(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        mcc = (tp*tn - fp*fn) / (np.sqrt(  (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)  ) + 1e-8)
        return round(mcc,3)
    
    
    @staticmethod
    def compute_balanced_acc(preds_prob, dtrain):
        y = dtrain.get_label()
        y_preds = preds_prob >= 0.5
        balanced_acc = balanced_accuracy_score(y,y_preds)
        #print('balanced_acc',balanced_acc)
        return 'balanced_accuracy', balanced_acc
        
    @staticmethod    
    def compute_metrics(y_preds, y_trues, task_type):
    
        if (task_type == 'classification'):
            acc = accuracy_score(y_trues, y_preds, normalize = True)
            conf_matrix = confusion_matrix(y_trues, y_preds)
            
            tn, fp, fn, tp = conf_matrix.ravel()
            precision = tp/(tp + fp)
            sensitivity = tp/(tp + fn)
            specificity = tn/(tn + fp)
            f1 = f1_score(y_trues, y_preds)
            mcc = Metrics.compute_mcc(y_trues, y_preds)
            roc_auc = roc_auc_score(y_trues, y_preds)
            balanced_accuracy = balanced_accuracy_score(y_trues, y_preds)    
            
            results = {'acc': acc, 'precision': precision,  'sensitivity': sensitivity, 'specificity': specificity, 'f1score': f1, 'mcc': mcc, 'roc_auc': roc_auc,
                        'tp_fp_fn_tp': conf_matrix.ravel(), 'confusion_matrix': conf_matrix}
        
        return results


# #### Parameter space

# In[7]:


class ParamFactory():
    
    _instance = None
    
    def __init__(self):        
        if ParamFactory._instance == None:
            self.setup_init_params()
            self.setup_random_seach_param_space()
            self.setup_bayesian_opt_param_space()
            ParamFactory._instance = self
        
                
    def get_init_params(self, modelname):
        return ParamFactory._instance.init_params[modelname]
        
    def get_random_seach_param_space(self, modelname):
        return ParamFactory._instance.random_seach_param_space[modelname]
    
    def get_bayesian_opt_param_space(self, modelname):
        return ParamFactory._instance.bayesian_opt_param_space[modelname]
        
    def setup_init_params(self):
        self.init_params = {}
        self.init_params['xgboost'] = { 'seed':  GLOBAL_RANDOM_STATE, 'use_label_encoder': False }
        self.init_params['adaboost'] = {'random_state': GLOBAL_RANDOM_STATE}#, 'n_iter_no_change': 30}#, 'algorithm':'SAMME'}
        self.init_params['catboost'] = { 'verbose': False,'random_state': GLOBAL_RANDOM_STATE} # 'task_type': "GPU" throws error with hyperopt        
        self.init_params['gbm'] = {'random_state': GLOBAL_RANDOM_STATE} # max_depth=1,  'n_estimators'=100, learning_rate=1.0, 
        self.init_params['randomforest'] = {'random_state' : GLOBAL_RANDOM_STATE, 'class_weight' : 'balanced' } 
        self.init_params['decisiontree'] = {'random_state' : GLOBAL_RANDOM_STATE, 'class_weight' : 'balanced' } 
        self.init_params['lda'] = {'random_state' : GLOBAL_RANDOM_STATE, 'class_weight' : 'balanced' } 
    
        
    def setup_bayesian_opt_param_space(self):
        self.bayesian_opt_param_space = dict()        
        self.bayesian_opt_param_space['xgboost'] = {
            'max_depth': hp.choice('max_depth', np.arange(3, 25+1, dtype=int)),
            'gamma': hp.quniform ('gamma', .5,5, .5),
            'alpha': hp.quniform ('alpha', 2, 9, 2),
            
        }
                
        self.bayesian_opt_param_space['adaboost'] = {
            'learning_rate': hp.loguniform('learning_rate', -2 ,0.),
            "n_estimators": hp.quniform('n_estimators', 50,300, 10),
        }
        
        self.bayesian_opt_param_space['catboost'] = {        
            'iterations': hp.quniform('iterations',10, 1000, 10),
            'depth': hp.quniform('depth',1, 8, 8),
            'learning_rate':  hp.loguniform('learning_rate',-2., 0.),

        }
        
        self.bayesian_opt_param_space['gbm'] = {
            "n_estimators": hp.quniform('n_estimators', 50, 500, 10), 
            'max_depth': hp.quniform('max_depth', 1., 5., 1),
            'learning_rate': hp.loguniform('learning_rate',-2., 0.),
            
        }
        
        self.bayesian_opt_param_space['randomforest'] = {
             'max_depth': hp.quniform('max_depth', 10, 100+1, 10),#np.array(np.arange(10, 100+1, 10, dtype=int).tolist()+[None])),
             'max_features': hp.quniform('max_features', 0.1, 1.0, .1),#hp.choice('max_features', ['log2', 'sqrt'] ),
             'min_samples_leaf': hp.quniform('min_samples_leaf', 0.1, 0.5, .1),
        }
        
        self.bayesian_opt_param_space['decisiontree'] = {
             'max_depth': hp.quniform('max_depth', 10, 100+1, 10),#np.array(np.arange(10, 100+1, 10, dtype=int).tolist()+[None])),
             'max_features': hp.quniform('max_features', 0.1, 1.0, .1),#hp.choice('max_features', ['log2', 'sqrt'] ),
             'min_samples_leaf': hp.quniform('min_samples_leaf', 0.1, 0.5, .1),
        }
        


# #### Model Generator




class ModelFactory():
    
    _instance = None
    
    def __init__(self):
        self.modelname2class = dict()
        self.modelname2class['xgboost'] = xgb.XGBClassifier
        self.modelname2class['adaboost'] = AdaBoostClassifier
        self.modelname2class['catboost'] = catboost.CatBoostClassifier
        self.modelname2class['gbm'] = GradientBoostingClassifier
        self.modelname2class['randomforest'] = RandomForestClassifier
        self.modelname2class['decisiontree'] = DecisionTreeClassifier
        self.modelname2class['lda'] = LinearDiscriminantAnalysis
        self.modelname2class['baggingclf'] = BaggingClassifier
        self.modelname2class['svc'] = svm.SVC
        self.modelname2class['svc'] = GaussianNB        
        
        if(ModelFactory._instance is None):
            ModelFactory._instance = self            

    #classmethod
    def getmodelclass(cls, modelname):        
        return cls._instance.modelname2class[modelname]


# #### Optimizer base class



class Optimizer:
    def __init__(self, model_name, scale_pos_weight = 0.5, num_iter = 5, metric = None):
        
        self.modelname = model_name
        self.modelclass = ModelFactory().getmodelclass(self.modelname)
        self.initparams = ParamFactory().get_init_params(self.modelname)
                
        self.scale_pos_weight = scale_pos_weight 
        self.num_iter = num_iter 
        self.metric = metric or log_loss
        
        
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None 
        
    def set_traindata(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train 
        
    def set_valdata(self, X_val, y_val):
        self.X_val, self.y_val = X_val, y_val  


# #### Bayesian Optimizer for Hyperparameter search



class HPOptimizer_Bayes(Optimizer):


    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)
        self.param_space = ParamFactory().get_bayesian_opt_param_space(self.modelname)
        
    def optimize(self):
    
        if X_train is None or y_train is None:
            raise Exception('Need dataset to tune hypter parameter for XGBoost')
            
        def objective(params):
        
            #print(params)            
            if (self.modelname == 'xgboost'):
 
                data_dmat = xgb.DMatrix(data=self.X_train, label=self.y_train)
                
                # xgboost cv
                cv_result = xgb.cv(
                    params=params,
                    dtrain=data_dmat,
                    #num_boost_round= 1200, # NOTE: number of estimator is set here
                    folds=KFold(n_splits=n_innersplits, shuffle = True, random_state=GLOBAL_RANDOM_STATE),
                
                    metrics= 'logloss',#'auc'
                    obj=None,
                    maximize=False, # maximize the metric spcified in feval
                    early_stopping_rounds=10, # NOTE: set to None to simplify debug
                    fpreproc=None,
                    as_pandas=True,
                    verbose_eval=False,
                    show_stdv=False,
                    seed=GLOBAL_RANDOM_STATE,
                    callbacks=None,
                    # shuffle=True, # NOTE: should be overwritten by folds
                )  
                
                score = np.nanmean(cv_result['test-logloss-mean'].values)
            else:

                params.update(self.initparams)
                if 'n_estimators' in params.keys():
                    params['n_estimators'] = int(params['n_estimators'])
                self.clf = self.modelclass(**params)
                

                cv_result = cross_val_score(self.clf, X_train, y_train, cv = KFold(n_splits = n_innersplits, shuffle = True, random_state= GLOBAL_RANDOM_STATE), scoring='neg_log_loss')
                
                #pdb.set_trace()
                score = cv_result.mean()
       
            return {'loss': score, 'status': STATUS_OK}
        
        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                                space = self.param_space,
                                algo = tpe.suggest,
                                max_evals = self.num_iter,
                                trials = trials)
                                
        best_hyperparams.update(self.initparams)
        
        if (self.modelname == 'xgboost'):
            best_hyperparams.update({'scale_pos_weight': self.scale_pos_weight})
        print(best_hyperparams)
        if 'n_estimators' in best_hyperparams.keys():
            best_hyperparams['n_estimators'] = int(best_hyperparams['n_estimators'])
                
        clf = self.modelclass(**best_hyperparams)
        print(clf)
        clf.fit(self.X_train, self.y_train)#, eval_metric = log_loss)
        

        
        return best_hyperparams, clf
        

# #### Classifiers



classifier_names = ['decisiontree']
classifier_name = classifier_names[0]

HP_optimizers= dict()
HP_optimizers[classifier_name] = HPOptimizer_Bayes(model_name = classifier_name, scale_pos_weight = 6.35, num_iter = 100, metric = log_loss )# = class_ratio_neg2pos)


print(classifier_names)


# #### Experiment configuration

# In[14]:


n_outersplits = 5
n_innersplits = 5
optimizer = 'bayesian_opt' # 'random_search'


# #### Driver code

# In[15]:


set_seed(GLOBAL_RANDOM_STATE)
setup_pretty_printing()
X, y, feature_names = DataLoader.load_real_dataset("./FeatureVector_Freezed_Final_NoBMD.csv", cols_to_drop=[], target_col='class')


# #### Training, Validation, and Testing

# In[ ]:



outer_fold = StratifiedKFold(n_outersplits, shuffle= True, random_state= GLOBAL_RANDOM_STATE)

metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','F1_Score','MCC','AUC_ROC']
result_dfs = pd.DataFrame(columns = ['Fold']+[metric+'_'+dstype for dstype in ['train','val','test'] for metric in metrics ]+['Time Taken (seconds)'])

y = y.flatten()

for fold_idx, (trainval_idx, test_idx) in enumerate(outer_fold.split(X, y)):
    
    X_trainval, X_test = X[trainval_idx,:], X[test_idx,:]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=GLOBAL_RANDOM_STATE)
    train_index, test_index = zip(*list(sss.split(X_trainval, y_trainval)))
    train_index, test_index = train_index[0], test_index[0]
    X_train, X_val, y_train, y_val = X_trainval[train_index], X_trainval[test_index], y_trainval[train_index], y_trainval[test_index] 
    
    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_train_norm = std_scaler.transform(X_train)
    X_val_norm = std_scaler.transform(X_val)
    X_test_norm = std_scaler.transform(X_test)

    for clf_idx, clf_name in enumerate(classifier_names):
        start_time = time.time()

        print(f'Training {clf_name}')
        clf_optimizer = HP_optimizers[clf_name] 
        clf_optimizer.set_traindata(X_train_norm, y_train)
        clf_optimizer.set_valdata(X_val_norm, y_val)
        
        print(f'Tuning hyper parameters for: {clf_name}')
        best_hyperparams, clf = clf_optimizer.optimize()
            
        print('re-fitted on train data using best hparams:', best_hyperparams)        
        
        print('Computing training metrics')            
        y_train_preds = clf.predict(X_train_norm)
        computed_metrics_train = Metrics.compute_metrics( y_train_preds, y_train, 'classification')
        print(computed_metrics_train['confusion_matrix'])
        
        
        print('Validating')            
        y_val_preds = clf.predict(X_val_norm)
        computed_metrics_val = Metrics.compute_metrics( y_val_preds, y_val, 'classification')
        print(computed_metrics_val['confusion_matrix'])
        
        print('Testing')      
        y_test_preds = clf.predict(X_test_norm)
        computed_metrics_test = Metrics.compute_metrics( y_test_preds, y_test, 'classification')        
        print(computed_metrics_test['confusion_matrix']) 

        
        end_time = time.time()
        time_taken = end_time - start_time

        result_dfs.loc[len(result_dfs.index)] = [fold_idx+1, computed_metrics_train['acc'], computed_metrics_train['precision'], 
                                                             computed_metrics_train['sensitivity'], computed_metrics_train['specificity'], computed_metrics_train['f1score'], 
                                                             computed_metrics_train['mcc'], computed_metrics_train['roc_auc'],
                                                             computed_metrics_val['acc'], computed_metrics_val['precision'], 
                                                             computed_metrics_val['sensitivity'], computed_metrics_val['specificity'], computed_metrics_val['f1score'], 
                                                             computed_metrics_val['mcc'], computed_metrics_val['roc_auc'], 
                                                             computed_metrics_test['acc'], computed_metrics_test['precision'], 
                                                             computed_metrics_test['sensitivity'], computed_metrics_test['specificity'], computed_metrics_test['f1score'], 
                                                             computed_metrics_test['mcc'], computed_metrics_test['roc_auc'], time_taken] 
        print('============================================================================================================================')
        print(f'Time taken by {clf_name}:{time_taken} seconds')
        print('============================================================================================================================')
        print()
        print()
        
    display(result_dfs)
    print()
    print()
    
    
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('                                     Training completed.                                                         ')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                
display(result_dfs)
print()
print()
result_dfs.to_csv('results_gpu.csv')


# #### Results



result_dfs.mean(axis = 0)

