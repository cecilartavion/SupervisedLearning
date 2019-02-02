
import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helper import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import time
     
#path = 'C:/Users/cecil/Dropbox/Dalton State College/georgia_tech/CSE7641/supervisor_learning/'
path = 'C:/Users/jasplund/Dropbox/Dalton State College/georgia_tech/CSE7641/supervisor_learning/'
df_spam = pd.read_csv(path + 'spam_data.txt', sep=',', header=None)
df_spam
df_adult_train = pd.read_csv(path + 'adult.data', sep=',', header=None)
df_adult_test = pd.read_csv(path + 'adult.data', sep=',', header=None)
df_adult_train.columns = ['age','workclass','fnlwgt','education','education_num',
                         'marital_status','occupation','relationship','race',
                         'sex','capital_gain','capital_loss','hours_per_week',
                         'native_country','income']
df_adult_test.columns = ['age','workclass','fnlwgt','education','education_num',
                         'marital_status','occupation','relationship','race',
                         'sex','capital_gain','capital_loss','hours_per_week',
                         'native_country','income']
adult_df = pd.concat([df_adult_train,df_adult_test],axis=0)
vals = pd.get_dummies(adult_df)
vals = vals.drop('income_ <=50K',1)
vals.columns = np.append(vals.columns.values[:-1], ['income'])



adultX = vals.drop('income',1).copy().values
adultX = adultX.astype(float)
adultY = vals['income'].copy().values
adultY = adultY.astype(float)


wine = pd.read_csv(path+'winequality-white.csv',sep=';')
wineX = wine.drop('quality',1).copy().values
wineY = wine['quality']

wineY[wineY.isin([1,2,3,4,5,6])] = 1
wineY[wineY.isin([7,8,9,10])] = 0

wineY = np.array(wineY)

d = adultX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
d = wineX.shape[1]
hiddens_wine = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]

adult_trnX, adult_tstX, adult_trnY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
wine_trnX, wine_tstX, wine_trnY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)
   
N_small_adult = small_adult_trnX.shape[0]
N_adult = adult_trnX.shape[0]
N_wine = wine_trnX.shape[0]



pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('KNN',knnC())])  

pipeW = Pipeline([('Scale',StandardScaler()),                
                 ('KNN',knnC())])  


params_wine= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,5),'KNN__weights':['uniform','distance']}
params_adult= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,5),'KNN__weights':['uniform','distance']}
#  
start = time.time()
wine_clf = basicResults(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,params_wine,'KNN','wine')        
end = time.time()
print(end-start)

start = time.time()
adult_clf = basicResults(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,params_adult,'KNN','adult')        
end = time.time()
print(end-start)

start = time.time()
wine_final_params=wine_clf.best_params_
adult_final_params=adult_clf.best_params_



pipeW.set_params(**wine_final_params)
makeTimingCurve(wineX,wineY,pipeW,'KNN','wine')
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'KNN','adult')
print(end-start)