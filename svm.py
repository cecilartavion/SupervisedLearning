import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from helper import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import time
from random import sample

class primalSVM_RBF(BaseEstimator, ClassifierMixin):
    '''http://scikit-learn.org/stable/developers/contributing.html'''
    
    def __init__(self, alpha=1e-9,gamma_frac=0.1,n_iter=2000):
         self.alpha = alpha
         self.gamma_frac = gamma_frac
         self.n_iter = n_iter
         
    def fit(self, X, y):
         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         
         # Get the kernel matrix
         dist = euclidean_distances(X,squared=True)
         median = np.median(dist) 
         del dist
         gamma = median
         gamma *= self.gamma_frac
         self.gamma = 1/gamma
         kernels = rbf_kernel(X,None,self.gamma )
         
         self.X_ = X
         self.classes_ = unique_labels(y)
         self.kernels_ = kernels
         self.y_ = y
         self.clf = SGDClassifier(loss='hinge',penalty='l2',alpha=self.alpha,
                                  l1_ratio=0,fit_intercept=True,verbose=False,
                                  average=False,learning_rate='optimal',
                                  class_weight='balanced',n_iter=self.n_iter,
                                  random_state=55)         
         self.clf.fit(self.kernels_,self.y_)
         
         # Return the classifier
         return self

    def predict(self, X):
         # Check is fit had been called
         check_is_fitted(self, ['X_', 'y_','clf','kernels_'])
         # Input validation
         X = check_array(X)
         new_kernels = rbf_kernel(X,self.X_,self.gamma )
         pred = self.clf.predict(new_kernels)
         return pred
     
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

sample_idx0 = sample(range(len(vals)),16000)
medium_adult = vals.iloc[sample_idx0]


adultX = medium_adult.drop('income',1).copy().values
adultX = adultX.astype(float)
adultY = medium_adult['income'].copy().values
adultY = adultY.astype(float)

sample_idx = sample(range(len(adultX)),4000)
small_adultX = adultX[sample_idx]
small_adultY = adultY[sample_idx]

wine = pd.read_csv(path+'winequality-white.csv',sep=';')
wineX = wine.drop('quality',1).copy().values
wineY = wine['quality']

wineY[wineY.isin([1,2,3,4,5,6])] = 1
wineY[wineY.isin([7,8,9,10])] = 0

wineY = np.array(wineY)
adult_trnX, adult_tstX, adult_trnY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
small_adult_trnX, small_adult_tstX, small_adult_trnY, small_adult_tstY = ms.train_test_split(small_adultX, small_adultY, test_size=0.3, random_state=0,stratify=small_adultY)     
wine_trnX, wine_tstX, wine_trnY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)
   
N_small_adult = small_adult_trnX.shape[0]
N_adult = adult_trnX.shape[0]
N_wine = wine_trnX.shape[0]

alphas = [10**-x for x in np.arange(1,9.01,1)]

gamma_fracsA = np.arange(0.2,2.1,0.5)
gamma_fracsW = np.arange(0.05,1.01,0.25)

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('SVM',primalSVM_RBF())])

pipeW = Pipeline([('Scale',StandardScaler()),
                 ('SVM',primalSVM_RBF())])


params_adult = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_adult)/.8)+1],'SVM__gamma_frac':gamma_fracsA}         
params_small_adult = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_small_adult)/.8)+1],'SVM__gamma_frac':gamma_fracsA}
params_wine = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_wine)/.8)+1],'SVM__gamma_frac':gamma_fracsW}
#  
start = time.time()
wine_clf = basicResults(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,params_wine,'SVM_RBF','wine')
end = time.time()
print(end-start)


start = time.time()
#This code was specially executed to avoid issues with the learning curve missing
# training and test size being so small that they only contained one class.
# To correct for this, we removed the tests when only 50 or 100 instances were
# used. 
import sklearn.model_selection as ms
import pandas as pd
from collections import defaultdict
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier as dtclf
def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced',truth)
    return accuracy_score(truth,pred,sample_weight=wts)
scorer = make_scorer(balanced_accuracy)    
clf_type = 'SVM_RBF'
dataset = 'small_adult'
cv = ms.GridSearchCV(pipeA,n_jobs=1,param_grid=params_small_adult,refit=True,verbose=10,cv=5,scoring=scorer)
print('2')
cv.fit(small_adult_trnX,small_adult_trnY)
print('3')
regTable = pd.DataFrame(cv.cv_results_)
print('4')
regTable.to_csv('./output/{}_{}_reg.csv'.format(clf_type,dataset),index=False)
print('5')
test_score = cv.score(small_adult_trnX,small_adult_trnY)
print('6')
with open('./output/test results.csv','a') as f:
    f.write('{},{},{},{}\n'.format(clf_type,dataset,test_score,cv.best_params_))    
print('7')
N = small_adult_trnX.shape[0]    
curve = ms.learning_curve(cv.best_estimator_,small_adult_trnX,small_adult_trnY,cv=5,train_sizes=[int(N*x/10) for x in range(1,8)],verbose=10,scoring=scorer)
print('8')
curve_train_scores = pd.DataFrame(index = curve[0],data = curve[1])
curve_test_scores  = pd.DataFrame(index = curve[0],data = curve[2])
curve_train_scores.to_csv('./output/{}_{}_LC_train.csv'.format(clf_type,dataset))
curve_test_scores.to_csv('./output/{}_{}_LC_test.csv'.format(clf_type,dataset))
end = time.time()
print(end-start)
adult_clf = cv

start = time.time()
wine_final_params = wine_clf.best_params_
wine_OF_params = wine_final_params.copy()
wine_OF_params['SVM__alpha'] = 1e-16
adult_final_params =adult_clf.best_params_
adult_OF_params = adult_final_params.copy()
adult_OF_params['SVM__alpha'] = 1e-16

pipeW.set_params(**wine_final_params)                     
makeTimingCurve(wineX,wineY,pipeW,'SVM_RBF','wine')
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeW,'SVM_RBF','adult')


pipeW.set_params(**wine_final_params)
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_RBF','wine')        
pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','adult')                

pipeA.set_params(**adult_OF_params)
iterationLC(pipeA,adult_trnX,adult_trnY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','adult')                
pipeW.set_params(**wine_OF_params)
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_RBF_OF','wine')                
end = time.time()
print(end-start)