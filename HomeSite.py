#script 1
#import libraries
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


# For Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

#read test and train sets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#explore the data
train.info()
train.head()


#dealing with missing values for train and test set

train.fillna(-1,inplace = True)
test.fillna(-1,inplace = True)


Y = train["QuoteConversion_Flag"]
golden_feature=[("CoverageField1B","PropertyField21B"),
                ("GeographicField6A","GeographicField8A"),
                ("GeographicField6A","GeographicField13A"),
                ("GeographicField8A","GeographicField13A"),
                ("GeographicField11A","GeographicField13A"),
                ("GeographicField8A","GeographicField11A")]
                
for featureA,featureB in golden_feature:
        train["_".join([featureA,featureB,"diff"])]=train[featureA]-train[featureB]
        test["_".join([featureA,featureB,"diff"])]=test[featureA]-test[featureB]
X = train.drop(["QuoteNumber","QuoteConversion_Flag",'CoverageField4A','CoverageField4B','GeographicField17A','GeographicField25B','GeographicField52B','GeographicField8B','PersonalField29','PersonalField38','PersonalField42','PersonalField5','PersonalField59','PersonalField62','PropertyField10','SalesField15','SalesField9','PersonalField72','PersonalField60',"PropertyField8","PropertyField3","PropertyField23","PropertyField14","PropertyField12",'PersonalField79','PersonalField74','PersonalField61','GeographicField19A','GeographicField22B','GeographicField24B','PersonalField19','PersonalField33','PersonalField52','PersonalField28','PersonalField32','PersonalField34','PersonalField44','PersonalField47','PersonalField50','PersonalField54','PersonalField56','GeographicField52A','GeographicField64','PersonalField22','PersonalField37','PersonalField45','PersonalField55','PersonalField83','PropertyField15','SalesField14','GeographicField12B','GeographicField14B','GeographicField16A','GeographicField40A','GeographicField63','GeographicField23B','GeographicField25A','GeographicField26A','GeographicField30A','PersonalField24','PersonalField78','PropertyField1A','PropertyField21A','PersonalField31','PersonalField35','PersonalField36','PersonalField46','PersonalField48','PersonalField49','PersonalField51','PersonalField59','PersonalField6','PersonalField75','PersonalField76','GeographicField13B','GeographicField5A','PersonalField53','PersonalField57','PropertyField17','PropertyField30','CoverageField2A','CoverageField2B','GeographicField12A','GeographicField16B','GeographicField21B','GeographicField24A','GeographicField27A','GeographicField27B','GeographicField54A','GeographicField9A','GeographicField9B','PersonalField30','PropertyField28','PropertyField4','GeographicField15A','GeographicField26B','GeographicField28B','GeographicField58A','GeographicField58B','PersonalField23','PersonalField63','PersonalField81','SalesField13'],axis=1)


               
test = test.drop(["QuoteNumber",'CoverageField4A','CoverageField4B','GeographicField17A','GeographicField25B','GeographicField52B','GeographicField8B','PersonalField29','PersonalField38','PersonalField42','PersonalField5','PersonalField59','PersonalField62','PropertyField10','SalesField15','SalesField9','PersonalField72','PersonalField60',"PropertyField8","PropertyField3","PropertyField23","PropertyField14","PropertyField12",'PersonalField79','PersonalField74','PersonalField61','GeographicField19A','GeographicField22B','GeographicField24B','PersonalField19','PersonalField33','PersonalField52','PersonalField28','PersonalField32','PersonalField34','PersonalField44','PersonalField47','PersonalField50','PersonalField54','PersonalField56','GeographicField52A','GeographicField64','PersonalField22','PersonalField37','PersonalField45','PersonalField55','PersonalField83','PropertyField15','SalesField14','GeographicField12B','GeographicField14B','GeographicField16A','GeographicField40A','GeographicField63','GeographicField23B','GeographicField25A','GeographicField26A','GeographicField30A','PersonalField24','PersonalField78','PropertyField1A','PropertyField21A','PersonalField31','PersonalField35','PersonalField36','PersonalField46','PersonalField48','PersonalField49','PersonalField51','PersonalField59','PersonalField6','PersonalField75','PersonalField76','GeographicField13B','GeographicField5A','PersonalField53','PersonalField57','PropertyField17','PropertyField30','CoverageField2A','CoverageField2B','GeographicField12A','GeographicField16B','GeographicField21B','GeographicField24A','GeographicField27A','GeographicField27B','GeographicField54A','GeographicField9A','GeographicField9B','PersonalField30','PropertyField28','PropertyField4','GeographicField15A','GeographicField26B','GeographicField28B','GeographicField58A','GeographicField58B','PersonalField23','PersonalField63','PersonalField81','SalesField13'],axis=1)
X["NAs"] = np.sum(X<0, axis = 1)
test["NAs"] = np.sum(test<0, axis = 1)

#handling dates
X['Date'] = pd.to_datetime(pd.Series(X['Original_Quote_Date']))
X = X.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

X['Year'] = X['Date'].apply(lambda x: int(str(x)[:4]))
X['Month'] = X['Date'].apply(lambda x: int(str(x)[5:7]))
X['weekday'] = X['Date'].dt.dayofweek

test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

X = X.drop('Date', axis=1)
test = test.drop('Date', axis=1)

#X.fillna(0,inplace = True)
#test.fillna(0,inplace = True) 



#converting strings to numeric or factorizing
for f in X.columns:
    if X[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X[f].values) + list(test[f].values))
        X[f] = lbl.transform(list(X[f].values))
        test[f] = lbl.transform(list(test[f].values))

#PCA Analysis
#from sklearn import decomposition
#pca = decomposition.PCA(n_components=70)
#pca.fit(X)
#X = pca.transform(X)

#randomForest Model
#from sklearn.ensemble import RandomForestClassifier
#forestParam = RandomForestClassifier(n_estimators = 500, max_depth = 50, min_samples_split = 1, random_state= 0)

#fit data
#ForestModel = forestParam.fit(X,Y)

#testing the score
#forestParam.score(X,Y)

#################################################################################
#xgboost model
import xgboost as xgb
from xgboost import XGBClassifier
seed = 10
clf = xgb.XGBClassifier(n_estimators=70,
                        nthread=-1,
                        max_depth=10,
                        learning_rate=0.125,
                        silent=True,
                        subsample=0.80,
                        colsample_bytree=0.80,
                        objective = "binary:logistic",
                        
                        )

from sklearn.cross_validation import KFold 

xgb_model = clf.fit(X, Y, eval_metric="auc")
#################################################################################
#cross validation

bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround)

#train test split
import sklearn
from sklearn import metrics
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X,Y)
xgb_model_cv = clf.fit(X_train, Y_train, eval_metric="auc")
#pred_train = xgb_model_cv.predict(X_train)
pred_test = xgb_model_cv.predict_proba(X_test)[:,1]
#train_score = xgb_model_cv.score(X_train, Y_train)
test_score = xgb_model_cv.score(X_test, Y_test)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, pred_test)
roc_auc = metrics.auc(fpr, tpr)
roc_auc

#################################################################################

#feature importance


parameters = {'nthread':-1, #when use hyperthread, xgboost may become slower
              'objective':'binary:logistic',
              'learning_rate': 0.11, #so called `eta` value
              'max_depth': 10,
             # 'min_child_weight': [11],
              'silent': 1,
              'subsample': 0.80,
              'colsample_bytree': 0.80,
              'n_estimators': 70, #number of trees, change it to 1000 for better results
              'seed': 9999}      
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X,Y)                          
dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test,label = Y_test)
num_round = 1000
evallist = [(dtest,'auc')]

xgb_model = xgb.train(parameters, dtrain,evals = evallist,early_stopping_rounds=20)
res_va = xgb_model.predict(dtest,ntree_limit=xgb_model.best_iteration) #prediction on validation set
print(metrics.roc_auc_score(Y_test, res_va) )  


test_preds = xgb_model.predict(dtest)
#train_score = xgb_model_cv.score(X_train, Y_train)
print(metrics.roc_auc_score(Y_test, test_preds) )  

xgb_model.get_fscore()      
                
xgb.plot_importance(xgb_model)

def importance_XGB(trained_model):
    impdf = []
    for ft, score in clf.booster().get_fscore().iteritems():
        impdf.append({'feature': ft, 'importance': score})
    impdf = pd.DataFrame(impdf)
    impdf = impdf.sort_values(by='importance', ascending=False).reset_index(drop=True)
    impdf['importance'] /= impdf['importance'].sum()
    return impdf
    
importance_XGB(xgb_model_cv)
#################################################################################

#GridSearch
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X,Y)

gbm_params = {
    'learning_rate': [0.025,0.125,0.11],
    'n_estimators': [20,40,60,70],
    'max_depth': [8,9,10,11,12],
    'subsample': [0.77,0.8,0.82],
    'colsample_bytree': [0.77,0.8,0.82]
}
gbm = xgb.XGBClassifier()
cv = StratifiedKFold(Y_train,n_folds = 5)
grid = GridSearchCV(gbm, gbm_params,scoring='roc_auc',cv=cv,verbose=10,n_jobs=-1)
gbmodel = grid.fit(X_train, Y_train)
print (grid.best_params_)
predictions = grid.best_estimator_.predict_proba(X_test)[:,1]
test_score = gbmodel.score(X_test, Y_test)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
roc_auc

#################################################################################

#prediction
#test = pca.transform(test)
class_predict = xgb_model.predict(test)
preds = clf.predict_proba(test)[:,1]
sample = pd.read_csv("sample_submission.csv")
sample.QuoteConversion_Flag = preds
sample.to_csv('xgbmodel_29thTry.csv', index=False)