import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helper import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
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

adult_trnX, adult_tstX, adult_trnY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
wine_trnX, wine_tstX, wine_trnY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)
   

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

pipeW = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])


d = adultX.shape[1]
d = int(d//(2**4))
d
wine_d = wineX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
hiddens_adult 
hiddens_wine = [(h,)*l for l in [1,2,3] for h in [wine_d,wine_d//2,wine_d*2]]
hiddens_wine 
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]
alphas
params_adult = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_adult}
params_wine = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_wine}

start = time.time()
adult_clf = basicResults(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,params_adult,'ANN','adult')        

adult_final_params =adult_clf.best_params_
adult_OF_params =adult_final_params.copy()
adult_OF_params 
adult_OF_params['MLP__alpha'] = 0
pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(adultX,adultY,pipeA,'ANN','adult')
pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900]},'ANN','adult')                
pipeA.set_params(**adult_OF_params)
pipeA.set_params(**{'MLP__early_stopping':False})               
iterationLC(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900]},'ANN_OF','adult')                
end = time.time()
print(end-start)

start = time.time()
wine_clf = basicResults(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,params_wine,'ANN','wine')        



wine_final_params =wine_clf.best_params_
wine_OF_params =wine_final_params.copy()
#wine_OF_params['MLP__alpha'] = 0
pipeW.set_params(**wine_final_params)
pipeW.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(wineX,wineY,pipeW,'ANN','wine')
pipeW.set_params(**wine_final_params)
pipeW.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900]},'ANN','wine')                
pipeW.set_params(**wine_OF_params)
pipeW.set_params(**{'MLP__early_stopping':False})               
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900]},'ANN_OF','wine')                
end = time.time()
print(end-start)