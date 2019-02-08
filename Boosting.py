import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helper import dtclf_pruned
import pandas as pd
import numpy as np
from helper import  basicResults,makeTimingCurve,iterationLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time


#path = 'C:/Users/cecil/Dropbox/Dalton State College/georgia_tech/CSE7641/supervisor_learning/'
path = 'C:/Users/jasplund/Dropbox/Dalton State College/georgia_tech/CSE7641/supervisor_learning/'

df_adult_train = pd.read_csv(path + 'adult.data', sep=',', header=None)
df_adult_test = pd.read_csv(path + 'adult.test', sep=',', header=None)
df_adult_train.columns = ['age','workclass','fnlwgt','education','education_num',
                         'marital_status','occupation','relationship','race',
                         'sex','capital_gain','capital_loss','hours_per_week',
                         'native_country','income']
df_adult_test.columns = ['age','workclass','fnlwgt','education','education_num',
                         'marital_status','occupation','relationship','race',
                         'sex','capital_gain','capital_loss','hours_per_week',
                         'native_country','income']
adult_df = pd.concat([df_adult_train,df_adult_test],axis=0)
adult_df.loc[adult_df['income']==' >50K.','income'] = ' >50K'
adult_df.loc[adult_df['income']==' <=50K.','income'] = ' <=50K'
vals = pd.get_dummies(adult_df)
vals = vals.drop('income_ <=50K',1)
vals.columns = np.append(vals.columns.values[:-1], ['income'])
vals = vals.drop(['relationship_ Husband','workclass_ ?'],axis=1)

adultX = vals.drop('income',1).copy().values
adultX = adultX.astype(float)
adultY = vals['income'].copy().values
adultY = adultY.astype(float)

wine = pd.read_csv(path+'winequality-white.csv',sep=';')
wineX = wine.drop('quality',1).copy().values
wineY = wine['quality']

wineY[wineY.isin([1,2,3,4,5,6])] = 1
wineY[wineY.isin([7,8,9,10])] = 0

alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
len(alphas)
adult_trnX, adult_tstX, adult_trnY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
wine_trnX, wine_tstX, wine_trnY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)
   

wine_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
adult_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                

paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45],
          'Boost__base_estimator__alpha':alphas}


paramsW = {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
           'Boost__base_estimator__alpha':alphas}
                                   
         
wine_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=wine_base,random_state=55)
adult_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=adult_base,random_state=55)
OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)

pipeW = Pipeline([('Scale',StandardScaler()),
                 ('Boost',wine_booster)])

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('Boost',adult_booster)])

#
start = time.time()
wine_clf = basicResults(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,paramsW,'Boost','wine')        
end = time.time()
print('Wine time:',end-start)

start = time.time()
adult_clf = basicResults(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,paramsA,'Boost','adult')        
end = time.time()
print('Adult time:',end-start)

#
#
#madelon_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
#adult_final_params = {'n_estimators': 10, 'learning_rate': 1}
#OF_params = {'learning_rate':1}

wine_final_params = wine_clf.best_params_
adult_final_params = adult_clf.best_params_
OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}

##
pipeW.set_params(**wine_final_params)
pipeA.set_params(**adult_final_params)
makeTimingCurve(wineX,wineY,pipeW,'Boost','wine')
makeTimingCurve(adultX,adultY,pipeA,'Boost','adult')
#
pipeW.set_params(**wine_final_params)
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost','wine')        
pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40]},'Boost','adult')                
pipeW.set_params(**OF_params)
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost_OF','wine')                
pipeA.set_params(**OF_params)
iterationLC(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40]},'Boost_OF','adult')                


               