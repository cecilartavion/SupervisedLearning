import sklearn.model_selection as ms
import pandas as pd
from helper import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('./output/DT_{}_nodecounts.csv'.format(dataset))
    
    return

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

#adult_trnX = vals_trn.drop('income_ >50K',1).copy().values
#adult_trnY = vals_trn['income_ >50K'].copy().values
#adult_tstX = vals_tst.drop('income_ >50K',1).copy().values
#adult_tstY = vals_tst['income_ >50K'].copy().values


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

alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]

adult_trnX, adult_tstX, adult_trnY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
wine_trnX, wine_tstX, wine_trnY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)
   
pipeA = Pipeline([('Scale',StandardScaler()),                 
                 ('DT',dtclf_pruned(random_state=55))])

pipeW = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('DT',dtclf_pruned(random_state=55))])
                                   
         
params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}

start = time.time()
wine_clf = basicResults(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,params,'DT','wine')        
end = time.time()
print(end-start)

start = time.time()
adult_clf = basicResults(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,params,'DT','adult')        
end = time.time()
print(end-start)


#wine_final_params = {'DT__alpha': -0.00031622776601683794, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}
#adult_final_params = {'class_weight': 'balanced', 'alpha': 0.0031622776601683794, 'criterion': 'entropy'}
wine_final_params = wine_clf.best_params_
adult_final_params = adult_clf.best_params_

pipeW.set_params(**wine_final_params)
makeTimingCurve(wineX,wineY,pipeW,'DT','wine')
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'DT','adult')


DTpruningVSnodes(pipeW,alphas,wine_trnX,wine_trnY,'wine')
DTpruningVSnodes(pipeA,alphas,adult_trnX,adult_trnY,'adult')               