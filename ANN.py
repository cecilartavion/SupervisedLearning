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

adult_trnX, adult_tstX, adult_trnY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
wine_trnX, wine_tstX, wine_trnY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)
   
#madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)     
#
#

#
#gdf_district = gpd.read_file(path+'MA_precincts_12_16.shp')
#df_district = gdf_district.drop('geometry',1).copy()
#df_district = df_district.drop(['District','Name','Ward','Pct','POP10','SEN12R',
#                                'SEN12D','SEN13D','SEN13R','SEN14D','SEN14R',
#                                'PRES16D','PRES16R','CD'],1).copy()
#df_district.columns
#df_district_numerical = pd.DataFrame(df_district.drop('City/Town',1).copy())
#df_district_numerical.convert_objects(convert_numeric=True)
#df_district_numerical['PRES12D'] = df_district_numerical['PRES12D'].str.replace(",","").astype(float)
#df_district_numerical['PRES12R'] = df_district_numerical['PRES12R'].str.replace(",","").astype(float)
#df_district_categorical = df_district['City/Town'].copy()
#df_district = pd.concat([df_district_numerical,pd.get_dummies(df_district_categorical)],axis=1)
#df_district['PRES12R_perc'] = df_district['PRES12R']/(df_district['PRES12D']+df_district['PRES12R'])
#df_district = df_district.drop(['PRES12R','PRES12D'],1)
#districtX = df_district.drop('PRES12R_perc',1).copy().values
#districtY = df_district['PRES12R_perc'].copy().values
#
#spamX = df_spam.drop(9,1).copy().values
#spamY = df_spam[9]
#spam_trgX, spam_tstX, spam_trgY, spam_tstY = ms.train_test_split(spamX, spamY, test_size=0.3, random_state=0)     
#district_trgX, district_tstX, district_trgY, district_tstY = ms.train_test_split(districtX, districtY, test_size=0.3, random_state=0)     
#rlf = svm.SVR(gamma=0.001)
#rlf.fit(district_trgX,district_trgY)
#
#clf = svm.SVC(gamma=0.001)
#clf.fit(spam_trgX,spam_trgY)
#
#import time
#
#start = time.time()
#
#clf = svm.SVC(gamma=0.001)
#clf.fit(adult_trgX[:10000],adult_trgY[:10000])
#
#end = time.time()
#
#print(end-start)
#
#N_adult = adult_trgX.shape[0]
#N_district = district_trgX.shape[0]
#
#alphas = [10**-x for x in np.arange(1,9.01,1/2)]


#Linear SVM
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

pipeW = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

#pipeM = Pipeline([('Scale',StandardScaler()),
#                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
#                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

d = adultX.shape[1]
d = int(d//(2**4))
d
wine_d = wineX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
hiddens_adult 
hiddens_wine = [(h,)*l for l in [1,2,3] for h in [wine_d,wine_d//2,wine_d*2]]
hiddens_wine 
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]

params_adult = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_adult}
params_wine = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_wine}

start = time.time()
adult_clf = basicResults(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,params_adult,'ANN','adult')        
end = time.time()
end-start

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

makeTimingCurve(adultX,adultY,pipeA,'ANN','adult')


start = time.time()
wine_clf = basicResults(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,params_wine,'ANN','wine')        
end = time.time()
end-start

wine_final_params =wine_clf.best_params_
wine_OF_params =wine_final_params.copy()
wine_OF_params 
wine_OF_params['MLP__alpha'] = 0
pipeW.set_params(**wine_final_params)
pipeW.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(wineX,wineY,pipeW,'ANN','wine')
pipeW.set_params(**wine_final_params)
pipeW.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900]},'ANN','wine')                
pipeW.set_params(**wine_OF_params)
pipeW.set_params(**{'MLP__early_stopping':False})               
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900]},'ANN_OF','wine')                

#
#params_adult = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_adult)/.8)+1]}
#params_district = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_district)/.8)+1]}
#
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#
#scores = ['precision', 'recall']
#
#from sklearn import datasets
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report
#from sklearn.svm import SVC
#
#
## Loading the Digits dataset
#digits = datasets.load_digits()
#
## To apply an classifier on this data, we need to flatten the image, to
## turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
#X = digits.images.reshape((n_samples, -1))
#y = digits.target
#
## Split the dataset in two equal parts
#X_train, X_test, y_train, y_test = train_test_split(
#    districtX, districtY, test_size=0.5, random_state=0)
#
## Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#
#scores = ['precision', 'recall']
#
#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()
#
#    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                       scoring='%s_macro' % score)
#    clf.fit(X_train, y_train)
#
#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = clf.cv_results_['mean_test_score']
#    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
#    print()
#
#    print("Detailed classification report:")
#    print()
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    print()
#    y_true, y_pred = y_test, clf.predict(X_test)
#    print(classification_report(y_true, y_pred))
#    print()                                        
#
#district_clf = basicResults(pipeM,district_trgX,district_trgY,district_tstX,district_tstY,params_district,'SVM_Lin','district')        
#adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'SVM_Lin','adult')        
#basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'SVM_RBF','adult')
##district_final_params = {'SVM__alpha': 0.031622776601683791, 'SVM__n_iter': 687.25}
#district_final_params = district_clf.best_params_
#district_OF_params = {'SVM__n_iter': 1303, 'SVM__alpha': 1e-16}
##adult_final_params ={'SVM__alpha': 0.001, 'SVM__n_iter': 54.75}
#adult_final_params =adult_clf.best_params_
#adult_OF_params ={'SVM__n_iter': 55, 'SVM__alpha': 1e-16}
#
#
#pipeM.set_params(**district_final_params)                     
#makeTimingCurve(districtX,districtY,pipeM,'SVM_Lin','district')
#pipeA.set_params(**adult_final_params)
#makeTimingCurve(adultX,adultY,pipeA,'SVM_Lin','adult')
#
#pipeM.set_params(**district_final_params)
#iterationLC(pipeM,district_trgX,district_trgY,district_tstX,district_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_Lin','district')        
#pipeA.set_params(**adult_final_params)
#iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_Lin','adult')                
#
#pipeA.set_params(**adult_OF_params)
#iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,200,5)},'SVM_LinOF','adult')                
#pipeM.set_params(**district_OF_params)
#iterationLC(pipeM,district_trgX,district_trgY,district_tstX,district_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_LinOF','district')                