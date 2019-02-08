import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import sklearn.model_selection as ms
from statsmodels import robust
import statistics as stats
import matplotlib.pyplot as plt

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
vals.columns
vals = vals.drop('income_ <=50K',1)

vals.columns = np.append(vals.columns.values[:-1], ['income'])

c = vals.corr().abs()

s = c.unstack()
so = s.sort_values(kind="quicksort")
so[so<1]
vals = vals.drop(['relationship_ Husband','workclass_ ?'],axis=1)

adultX = vals.drop('income',1).copy().values
adultX = adultX.astype(float)
adultY = vals['income'].copy().values
adultY = adultY.astype(float)

wine = pd.read_csv(path+'winequality-white.csv',sep=';')
wine['quality'].hist(bins=6)


#histogram plot for frequencies of the ratings
out = pd.cut(wine['quality'], bins=range(2,10), include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(7,5))
# print(wine)
ax.set_title('Wine Quality Frequency')
ax.set_xlabel('Rating')
ax.set_ylabel('Frequency')
ax.set_xticklabels([3,4,5,6,7,8,9])
rects = ax.patches
print(rects)
count,_ = np.histogram(wine['quality'], bins = range(3,11))
print(count)
# Make some labels.
labels = list(count)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
            ha='center', va='bottom')
fig = ax.get_figure()
#fig.savefig(path + 'hist_wine.pdf')

mad_score_wine = robust.mad(wine['quality'])
mad_score_wine 
wine[abs(wine['quality'].values-stats.median(wine['quality'].values))/mad_score_wine > 2]

mad_score_adult = robust.mad(vals['income'])
vals[abs(vals['income'].values-stats.median(vals['income'].values))/mad_score_adult > 2]

g = sns.heatmap(wine.corr(),annot=True, fmt='.2f') #Use heat map to show little colinearity.
# g = sns.pairplot(wine, hue="quality", palette="husl") #send to figure.

fig = g.get_figure()
#fig.savefig(path+ 'heat_map_wine.pdf')


#summary table
wine.describe().to_latex()
wine.describe()

len(wineY[wineY==1])/len(wineY)
adult_df.describe().to_latex()
adult_df.describe()
adult_df.describe(include='O').to_latex()
adult_df.describe(include='O')

np.unique(adult_df['income'].values)
adult_df['income'].value_counts().plot(kind='bar')
#There are no missing data.
wine.isnull().values.any()
adult_df.isnull().values.any()

wine.info()
adult_df.info()

#######################################
#######################################
#######################################
#######################################
#######################################

ann_wine_reg = pd.read_csv(path + 'output/ANN_wine_reg.csv',sep=',')
ann_wine_reg_fit_time = ann_wine_reg[['mean_fit_time','param_MLP__activation','param_MLP__alpha','param_MLP__hidden_layer_sizes']]

ann_adult_reg = pd.read_csv(path + 'output/ANN_adult_reg.csv',sep=',')
ann_adult_reg_fit_time = ann_adult_reg[['mean_fit_time','param_MLP__activation','param_MLP__alpha','param_MLP__hidden_layer_sizes']]


#######################################
#######################################
#######################################
#######################################
#######################################

alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
color_names = ['firebrick','seagreen','deepskyblue','red','blue','g','steelblue','olive','fuchsia']

ann_relu_reg = ann_wine_reg_fit_time[ann_wine_reg_fit_time['param_MLP__activation']=='relu'][['mean_fit_time','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_relu_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_relu_reg['mean_fit_time'].values[i::9]

y0 = ann_relu_reg['mean_fit_time'].values[0::9]
y1 = ann_relu_reg['mean_fit_time'].values[1::9]
y2 = ann_relu_reg['mean_fit_time'].values[2::9]
y3 = ann_relu_reg['mean_fit_time'].values[3::9]
y4 = ann_relu_reg['mean_fit_time'].values[4::9]
y5 = ann_relu_reg['mean_fit_time'].values[5::9]
y6 = ann_relu_reg['mean_fit_time'].values[6::9]
y7 = ann_relu_reg['mean_fit_time'].values[7::9]
y8 = ann_relu_reg['mean_fit_time'].values[8::9]
ax = plt.subplot(111)
X = np.arange(13)*2
plt.bar(X-0.6,y0,width=0.15,color=color_names[0],align='center')
plt.bar(X-0.45,y1,width=0.15,color=color_names[1],align='center')
plt.bar(X-0.3,y2,width=0.15,color=color_names[2],align='center')
plt.bar(X-0.15,y3,width=0.15,color=color_names[3],align='center')
plt.bar(X,y4,width=0.15,color=color_names[4],align='center')
plt.bar(X+0.2,y5,width=0.15,color=color_names[5],align='center')
plt.bar(X+0.4,y6,width=0.15,color=color_names[6],align='center')
plt.bar(X+0.6,y7,width=0.15,color=color_names[7],align='center')
plt.bar(X+0.8,y8,width=0.15,color=color_names[8],align='center')

plt.autoscale(tight=True)

#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
################## ANN ################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

ann_wine_reg = pd.read_csv(path + 'output/ANN_wine_reg.csv',sep=',')
ann_wine_reg_fit_time = ann_wine_reg[['mean_fit_time','param_MLP__activation','param_MLP__alpha','param_MLP__hidden_layer_sizes']]

ann_adult_reg = pd.read_csv('output/ANN_adult_reg.csv',sep=',')
ann_adult_reg_fit_time = ann_adult_reg[['mean_fit_time','param_MLP__activation','param_MLP__alpha','param_MLP__hidden_layer_sizes']]

#######################################
#######################################
#######################################
#######################################
#######################################

ann_relu_reg = ann_wine_reg[ann_wine_reg['param_MLP__activation']=='relu'][['mean_fit_time','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_relu_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_relu_reg['mean_fit_time'].values[i::9]
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_wine_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
#    axarr[i, j].set_ylim([0,1])
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_fit_mean_time_relu_wine.pdf')


#######################################
#######################################
#######################################
#######################################
#######################################

ann_relu_reg = ann_wine_reg[ann_wine_reg['param_MLP__activation']=='logistic'][['mean_fit_time','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_relu_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_relu_reg['mean_fit_time'].values[i::9]
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_wine_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
#    axarr[i, j].set_ylim([0,1])
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_fit_mean_time_log_wine.pdf')

####################################
####################################
####################################
####################################
####################################

ann_log_reg = ann_adult_reg[ann_adult_reg['param_MLP__activation']=='relu'][['mean_fit_time','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_log_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_log_reg['mean_fit_time'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_adult_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
#    axarr[i, j].set_ylim([0,1])
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_fit_mean_time_relu_adult.pdf')

####################################
####################################
####################################
####################################
####################################

ann_log_reg = ann_adult_reg[ann_adult_reg['param_MLP__activation']=='logistic'][['mean_fit_time','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_log_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_log_reg['mean_fit_time'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_adult_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
#    axarr[i, j].set_ylim([0,1])
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_fit_mean_time_log_adult.pdf')


#######################################
#######################################
#######################################
#######################################
#######################################

ann_relu_reg = ann_wine_reg[ann_wine_reg['param_MLP__activation']=='relu'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_relu_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_relu_reg['mean_train_score'].values[i::9]
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_wine_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_train_mean_score_relu_wine.pdf')
####################################
####################################
####################################
####################################
####################################

ann_log_reg = ann_wine_reg[ann_wine_reg['param_MLP__activation']=='logistic'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_log_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_log_reg['mean_train_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_wine_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_train_mean_score_log_wine.pdf')


#######################################
#######################################
#######################################
#######################################
#######################################

ann_relu_reg = ann_wine_reg[ann_wine_reg['param_MLP__activation']=='relu'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_relu_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_relu_reg['mean_test_score'].values[i::9]
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_wine_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_test_mean_score_relu_wine.pdf')
####################################
####################################
####################################
####################################
####################################

ann_log_reg = ann_wine_reg[ann_wine_reg['param_MLP__activation']=='logistic'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_log_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_log_reg['mean_test_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_wine_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_test_mean_score_log_wine.pdf')

####################################
####################################
####################################
####################################
####################################

ann_relu_reg = ann_adult_reg[ann_adult_reg['param_MLP__activation']=='relu'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_relu_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_relu_reg['mean_train_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))


hidden_layer_names = np.unique(ann_adult_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_train_mean_score_relu_adult.pdf')


####################################
####################################
####################################
####################################
####################################

ann_log_reg = ann_adult_reg[ann_adult_reg['param_MLP__activation']=='logistic'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_log_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_log_reg['mean_train_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_adult_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_train_mean_score_log_adult.pdf')


####################################
####################################
####################################
####################################
####################################

ann_relu_reg = ann_adult_reg[ann_adult_reg['param_MLP__activation']=='relu'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_relu_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_relu_reg['mean_train_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))


hidden_layer_names = np.unique(ann_adult_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_test_mean_score_relu_adult.pdf')


####################################
####################################
####################################
####################################
####################################

ann_log_reg = ann_adult_reg[ann_adult_reg['param_MLP__activation']=='logistic'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ann_log_reg['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ann_log_reg['mean_train_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ann_adult_reg['param_MLP__alpha']))
hidden_layer_names = np.unique(ann_adult_reg_fit_time['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')

for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ann_test_mean_score_log_adult.pdf')

####################################
####################################
####################################
####################################
####################################

ann_wine_train_LC = pd.read_csv('output/ANN_wine_LC_train.csv',sep=',')
ann_wine_test_LC = pd.read_csv('output/ANN_wine_LC_test.csv',sep=',')
ann_wine_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ann_wine_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ann_wine_train_LC = ann_wine_train_LC.set_index('training_sizes')
ann_wine_test_LC = ann_wine_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(ann_wine_train_LC,axis=1)
mean_test_size_score = np.mean(ann_wine_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("ann_wine_train_LC.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

ann_adult_train_LC = pd.read_csv('output/ANN_adult_LC_train.csv',sep=',')
ann_adult_test_LC = pd.read_csv('output/ANN_adult_LC_test.csv',sep=',')
ann_adult_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ann_adult_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ann_adult_train_LC = ann_adult_train_LC.set_index('training_sizes')
ann_adult_test_LC = ann_adult_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(ann_adult_train_LC,axis=1)
mean_test_size_score = np.mean(ann_adult_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Adult Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("ann_adult_train_LC.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

ann_iter_wine = pd.read_csv('output/ITER_base_ANN_wine.csv',sep=',')
ann_iter_wine_alpha_0 = pd.read_csv('output/ITER_base_ANN_OF_wine.csv',sep=',')
ann_iter_wine_train = ann_iter_wine[['mean_train_score','param_MLP__max_iter','std_train_score']].copy()
ann_iter_wine_test = ann_iter_wine[['mean_test_score','param_MLP__max_iter','std_test_score']].copy()
ann_iter_wine_alpha_0_train = ann_iter_wine_alpha_0[['mean_train_score','param_MLP__max_iter','std_train_score']].copy()
ann_iter_wine_alpha_0_test = ann_iter_wine_alpha_0[['mean_test_score','param_MLP__max_iter','std_test_score']].copy()

f = plt.figure()
plt.plot(list(ann_iter_wine_train['param_MLP__max_iter']), list(ann_iter_wine_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(ann_iter_wine_train['param_MLP__max_iter']), 
                 list(ann_iter_wine_train['mean_train_score'].values-ann_iter_wine_train['std_train_score']), 
                 list(ann_iter_wine_train['mean_train_score'].values+ann_iter_wine_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(ann_iter_wine_test['param_MLP__max_iter']), list(ann_iter_wine_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(ann_iter_wine_test['param_MLP__max_iter']), 
                 list(ann_iter_wine_test['mean_test_score'].values-ann_iter_wine_test['std_test_score']), 
                 list(ann_iter_wine_test['mean_test_score'].values+ann_iter_wine_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Max Number of Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

#f.savefig("ann_max_iter_wine.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################

f = plt.figure()
plt.plot(list(ann_iter_wine_alpha_0_train['param_MLP__max_iter']), list(ann_iter_wine_alpha_0_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(ann_iter_wine_alpha_0_train['param_MLP__max_iter']), 
                 list(ann_iter_wine_alpha_0_train['mean_train_score'].values-ann_iter_wine_alpha_0_train['std_train_score']), 
                 list(ann_iter_wine_alpha_0_train['mean_train_score'].values+ann_iter_wine_alpha_0_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(ann_iter_wine_alpha_0_test['param_MLP__max_iter']), list(ann_iter_wine_alpha_0_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(ann_iter_wine_alpha_0_test['param_MLP__max_iter']), 
                 list(ann_iter_wine_alpha_0_test['mean_test_score'].values-ann_iter_wine_alpha_0_test['std_test_score']), 
                 list(ann_iter_wine_alpha_0_test['mean_test_score'].values+ann_iter_wine_alpha_0_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Max Number of Iterations Alpha 0 vs. Accuracy',fontsize=20)
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

#f.savefig("ann_max_iter_alpha_0_wine.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################

ann_iter_adult = pd.read_csv('output/ITER_base_ANN_adult.csv',sep=',')
ann_iter_adult_alpha_0 = pd.read_csv('output/ITER_base_ANN_OF_adult.csv',sep=',')
ann_iter_adult_train = ann_iter_adult[['mean_train_score','param_MLP__max_iter','std_train_score']].copy()
ann_iter_adult_test = ann_iter_adult[['mean_test_score','param_MLP__max_iter','std_test_score']].copy()
ann_iter_adult_alpha_0_train = ann_iter_adult_alpha_0[['mean_train_score','param_MLP__max_iter','std_train_score']].copy()
ann_iter_adult_alpha_0_test = ann_iter_adult_alpha_0[['mean_test_score','param_MLP__max_iter','std_test_score']].copy()

f = plt.figure()
plt.plot(list(ann_iter_adult_train['param_MLP__max_iter']), list(ann_iter_adult_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(ann_iter_adult_train['param_MLP__max_iter']), 
                 list(ann_iter_adult_train['mean_train_score'].values-ann_iter_adult_train['std_train_score']), 
                 list(ann_iter_adult_train['mean_train_score'].values+ann_iter_adult_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(ann_iter_adult_test['param_MLP__max_iter']), list(ann_iter_adult_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(ann_iter_adult_test['param_MLP__max_iter']), 
                 list(ann_iter_adult_test['mean_test_score'].values-ann_iter_adult_test['std_test_score']), 
                 list(ann_iter_adult_test['mean_test_score'].values+ann_iter_adult_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Max Number of Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

#f.savefig("ann_max_iter_adult.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################

f = plt.figure()
plt.plot(list(ann_iter_adult_alpha_0_train['param_MLP__max_iter']), list(ann_iter_adult_alpha_0_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(ann_iter_adult_alpha_0_train['param_MLP__max_iter']), 
                 list(ann_iter_adult_alpha_0_train['mean_train_score'].values-ann_iter_adult_alpha_0_train['std_train_score']), 
                 list(ann_iter_adult_alpha_0_train['mean_train_score'].values+ann_iter_adult_alpha_0_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(ann_iter_adult_alpha_0_test['param_MLP__max_iter']), list(ann_iter_adult_alpha_0_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(ann_iter_adult_alpha_0_test['param_MLP__max_iter']), 
                 list(ann_iter_adult_alpha_0_test['mean_test_score'].values-ann_iter_adult_alpha_0_test['std_test_score']), 
                 list(ann_iter_adult_alpha_0_test['mean_test_score'].values+ann_iter_adult_alpha_0_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Max Number of Iterations Alpha 0 vs. Accuracy',fontsize=20)
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

#f.savefig("ann_max_iter_alpha_0_adult.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

ann_iter_wine = pd.read_csv('output/ITERtestSET_ANN_wine.csv',sep=',')
ann_iter_wine_alpha_0 = pd.read_csv('output/ITERtestSET_ANN_OF_wine.csv',sep=',')

f = plt.figure()
plt.plot(list(ann_iter_wine['param_MLP__max_iter']), list(ann_iter_wine['train acc']), 
         'k', color='r',label = 'Train')
plt.plot(list(ann_iter_wine['param_MLP__max_iter']), list(ann_iter_wine['test acc']), 
        'k', color='b', label='Test')
plt.plot(list(ann_iter_wine_alpha_0['param_MLP__max_iter']), list(ann_iter_wine_alpha_0['train acc']), 
         'k', color='m',label = 'Train Alpha 0')
plt.plot(list(ann_iter_wine_alpha_0['param_MLP__max_iter']), list(ann_iter_wine_alpha_0['test acc']), 
        'k', color='c', label='Test Alpha 0')
plt.title('Max Number of Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("ann_max_iter_wine_final.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

ann_iter_adult = pd.read_csv('output/ITERtestSET_ANN_adult.csv',sep=',')
ann_iter_adult_alpha_0 = pd.read_csv('output/ITERtestSET_ANN_OF_adult.csv',sep=',')

f = plt.figure()
plt.plot(list(ann_iter_adult['param_MLP__max_iter']), list(ann_iter_adult['train acc']), 
         'k', color='r',label = 'Train')
plt.plot(list(ann_iter_adult['param_MLP__max_iter']), list(ann_iter_adult['test acc']), 
        'k', color='b', label='Test')
plt.plot(list(ann_iter_adult_alpha_0['param_MLP__max_iter']), list(ann_iter_adult_alpha_0['train acc']), 
         'k', color='m',label = 'Train Alpha 0')
plt.plot(list(ann_iter_adult_alpha_0['param_MLP__max_iter']), list(ann_iter_adult_alpha_0['test acc']), 
        'k', color='c', label='Test Alpha 0')
plt.title('Max Number of Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("ann_max_iter_adult_final.pdf", bbox_inches='tight')



####################################
####################################
####################################
####################################
####################################
####################################
####################################
############## Boost ###############
####################################
####################################
####################################
####################################
####################################
####################################
####################################

boost_wine_reg = pd.read_csv(path + 'output/Boost_wine_reg.csv',sep=',')
boost_wine_reg_fit_time = boost_wine_reg[['mean_fit_time',
                                      'param_Boost__base_estimator__alpha',
                                      'param_Boost__n_estimators']]

boost_adult_reg = pd.read_csv(path + 'output/Boost_adult_reg.csv',sep=',')
boost_adult_reg_fit_time = boost_adult_reg[['mean_fit_time',
                                          'param_Boost__base_estimator__alpha',
                                          'param_Boost__n_estimators']]

####################################
####################################
####################################
####################################
####################################

x = boost_wine_reg['param_Boost__base_estimator__alpha']
y = np.zeros(10)
y = list(y)
for i in range(10):
  y[i] = boost_wine_reg['mean_fit_time'].values[i::10]

estimates = np.unique(boost_wine_reg['param_Boost__n_estimators'])
fig, axarr = plt.subplots(5, 2, figsize=(8, 12),sharex='col', sharey='row')
alphas = np.unique(boost_wine_reg['param_Boost__base_estimator__alpha'])

for i in range(2):
  for j in range(5):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(estimates[i+2*j],fontsize= 15)
#    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Complexity Time',fontsize= 20)
fig.text(0.04, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.2)
fig.get_figure()
fig
fig.savefig('boost_estimator_alpha_time_wine.pdf')

####################################
####################################
####################################
####################################
####################################

estimates = np.unique(boost_adult_reg['param_Boost__n_estimators'])
n_est = len(estimates)
x = boost_adult_reg['param_Boost__base_estimator__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = boost_adult_reg['mean_fit_time'].values[i::n_est]

fig, axarr = plt.subplots(4, 2, figsize=(8, 12),sharex='col', sharey='row')
alphas = np.unique(boost_adult_reg['param_Boost__base_estimator__alpha'])

for i in range(2):
  for j in range(4):
    if i+2*j>6:
        axarr[j, i].set_xticklabels(alphas)
        axarr[j, i].tick_params(axis='x', rotation=90)
        break
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(estimates[i+2*j],fontsize= 15)
#    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Complexity Time',fontsize= 20)
fig.text(0.04, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.2)
fig.get_figure()
fig
fig.savefig('boost_estimator_alpha_time_adult.pdf')


####################################
####################################
####################################
####################################
####################################

x = boost_wine_reg['param_Boost__base_estimator__alpha']
y = np.zeros(10)
y = list(y)
for i in range(10):
  y[i] = boost_wine_reg['mean_train_score'].values[i::10]

estimates = np.unique(boost_wine_reg['param_Boost__n_estimators'])
fig, axarr = plt.subplots(5, 2, figsize=(8, 12),sharex='col', sharey='row')
alphas = np.unique(boost_wine_reg['param_Boost__base_estimator__alpha'])

for i in range(2):
  for j in range(5):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(estimates[i+2*j],fontsize= 15)
    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Complexity Train Accuracy',fontsize= 20)
fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.2)
fig.get_figure()
fig
fig.savefig('boost_estimator_alpha_accuracy_train_wine.pdf')

####################################
####################################
####################################
####################################
####################################

x = boost_wine_reg['param_Boost__base_estimator__alpha']
y = np.zeros(10)
y = list(y)
for i in range(10):
  y[i] = boost_wine_reg['mean_test_score'].values[i::10]

estimates = np.unique(boost_wine_reg['param_Boost__n_estimators'])
fig, axarr = plt.subplots(5, 2, figsize=(8, 12),sharex='col', sharey='row')
alphas = np.unique(boost_wine_reg['param_Boost__base_estimator__alpha'])

for i in range(2):
  for j in range(5):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(estimates[i+2*j],fontsize= 15)
    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Complexity Test Accuracy',fontsize= 20)
fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.2)
fig.get_figure()
fig
fig.savefig('boost_estimator_alpha_accuracy_test_wine.pdf')

####################################
####################################
####################################
####################################
####################################

estimates = np.unique(boost_adult_reg['param_Boost__n_estimators'])
n_est = len(estimates)
x = boost_adult_reg['param_Boost__base_estimator__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = boost_adult_reg['mean_train_score'].values[i::n_est]

fig, axarr = plt.subplots(4, 2, figsize=(8, 12),sharex='col', sharey='row')
alphas = np.unique(boost_adult_reg['param_Boost__base_estimator__alpha'])

for i in range(2):
  for j in range(4):
    if i+2*j>6:
        axarr[j, i].set_xticklabels(alphas)
        axarr[j, i].tick_params(axis='x', rotation=90)
        break
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(estimates[i+2*j],fontsize= 15)
    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Complexity Train Accuracy',fontsize= 20)
fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.2)
fig.get_figure()
fig
#fig.savefig('boost_estimator_alpha_accuracy_train_adult.pdf')

####################################
####################################
####################################
####################################
####################################

estimates = np.unique(boost_adult_reg['param_Boost__n_estimators'])
n_est = len(estimates)
x = boost_adult_reg['param_Boost__base_estimator__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = boost_adult_reg['mean_test_score'].values[i::n_est]

fig, axarr = plt.subplots(4, 2, figsize=(8, 12),sharex='col', sharey='row')
alphas = np.unique(boost_adult_reg['param_Boost__base_estimator__alpha'])

for i in range(2):
  for j in range(4):
    if i+2*j>6:
        axarr[j, i].set_xticklabels(alphas)
        axarr[j, i].tick_params(axis='x', rotation=90)
        break
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(estimates[i+2*j],fontsize= 15)
    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Complexity Test Accuracy',fontsize= 20)
fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.2)
fig.get_figure()
fig
#fig.savefig('boost_estimator_alpha_accuracy_test_adult.pdf')

####################################
####################################
####################################
####################################
####################################

boost_wine_train_LC = pd.read_csv('output/Boost_wine_LC_train.csv',sep=',')
boost_wine_test_LC = pd.read_csv('output/Boost_wine_LC_test.csv',sep=',')
boost_wine_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
boost_wine_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
boost_wine_train_LC = boost_wine_train_LC.set_index('training_sizes')
boost_wine_test_LC = boost_wine_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(boost_wine_train_LC,axis=1)
mean_test_size_score = np.mean(boost_wine_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("boost_wine_train_LC.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

boost_adult_train_LC = pd.read_csv('output/Boost_adult_LC_train.csv',sep=',')
boost_adult_test_LC = pd.read_csv('output/Boost_adult_LC_test.csv',sep=',')
boost_adult_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
boost_adult_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
boost_adult_train_LC = boost_adult_train_LC.set_index('training_sizes')
boost_adult_test_LC = boost_adult_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(boost_adult_train_LC,axis=1)
mean_test_size_score = np.mean(boost_adult_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("boost_adult_train_LC.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

boost_iter_wine = pd.read_csv('output/ITER_base_Boost_wine.csv',sep=',')
boost_iter_wine_alpha_0 = pd.read_csv('output/ITER_base_Boost_OF_wine.csv',sep=',')
boost_iter_wine_train = boost_iter_wine[['mean_train_score','param_Boost__n_estimators','std_train_score']].copy()
boost_iter_wine_test = boost_iter_wine[['mean_test_score','param_Boost__n_estimators','std_test_score']].copy()
boost_iter_wine_alpha_0_train = boost_iter_wine_alpha_0[['mean_train_score','param_Boost__n_estimators','std_train_score']].copy()
boost_iter_wine_alpha_0_test = boost_iter_wine_alpha_0[['mean_test_score','param_Boost__n_estimators','std_test_score']].copy()

f = plt.figure()
plt.plot(list(boost_iter_wine_train['param_Boost__n_estimators']), list(boost_iter_wine_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(boost_iter_wine_train['param_Boost__n_estimators']), 
                 list(boost_iter_wine_train['mean_train_score'].values-boost_iter_wine_train['std_train_score']), 
                 list(boost_iter_wine_train['mean_train_score'].values+boost_iter_wine_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(boost_iter_wine_test['param_Boost__n_estimators']), list(boost_iter_wine_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(boost_iter_wine_test['param_Boost__n_estimators']), 
                 list(boost_iter_wine_test['mean_test_score'].values-boost_iter_wine_test['std_test_score']), 
                 list(boost_iter_wine_test['mean_test_score'].values+boost_iter_wine_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Number of Boost Estimators vs. Accuracy',fontsize=20)
plt.xlabel('Number of Estimators',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

#f.savefig("boost_num_estimators_wine_first_run.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################

f = plt.figure()
plt.plot(list(boost_iter_wine_alpha_0_train['param_Boost__n_estimators']), list(boost_iter_wine_alpha_0_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(boost_iter_wine_alpha_0_train['param_Boost__n_estimators']), 
                 list(boost_iter_wine_alpha_0_train['mean_train_score'].values-boost_iter_wine_alpha_0_train['std_train_score']), 
                 list(boost_iter_wine_alpha_0_train['mean_train_score'].values+boost_iter_wine_alpha_0_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(boost_iter_wine_alpha_0_test['param_Boost__n_estimators']), list(boost_iter_wine_alpha_0_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(boost_iter_wine_alpha_0_test['param_Boost__n_estimators']), 
                 list(boost_iter_wine_alpha_0_test['mean_test_score'].values-boost_iter_wine_alpha_0_test['std_test_score']), 
                 list(boost_iter_wine_alpha_0_test['mean_test_score'].values+boost_iter_wine_alpha_0_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Num. of Boost Estimators Complexity -1 vs. Acc.',fontsize=20)
plt.xlabel('Number of Estimators',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

#f.savefig("boost_num_estimators_wine_alpha_0_first_run.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################

boost_iter_adult = pd.read_csv('output/ITER_base_Boost_adult.csv',sep=',')
boost_iter_adult_alpha_0 = pd.read_csv('output/ITER_base_Boost_OF_adult.csv',sep=',')
boost_iter_adult_train = boost_iter_adult[['mean_train_score','param_Boost__n_estimators','std_train_score']].copy()
boost_iter_adult_test = boost_iter_adult[['mean_test_score','param_Boost__n_estimators','std_test_score']].copy()
boost_iter_adult_alpha_0_train = boost_iter_adult_alpha_0[['mean_train_score','param_Boost__n_estimators','std_train_score']].copy()
boost_iter_adult_alpha_0_test = boost_iter_adult_alpha_0[['mean_test_score','param_Boost__n_estimators','std_test_score']].copy()

f = plt.figure()
plt.plot(list(boost_iter_adult_train['param_Boost__n_estimators']), list(boost_iter_adult_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(boost_iter_adult_train['param_Boost__n_estimators']), 
                 list(boost_iter_adult_train['mean_train_score'].values-boost_iter_adult_train['std_train_score']), 
                 list(boost_iter_adult_train['mean_train_score'].values+boost_iter_adult_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(boost_iter_adult_test['param_Boost__n_estimators']), list(boost_iter_adult_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(boost_iter_adult_test['param_Boost__n_estimators']), 
                 list(boost_iter_adult_test['mean_test_score'].values-boost_iter_adult_test['std_test_score']), 
                 list(boost_iter_adult_test['mean_test_score'].values+boost_iter_adult_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Number of Boost Estimators vs. Accuracy',fontsize=20)
plt.xlabel('Number of Estimators',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

#f.savefig("boost_num_estimators_adult_first_run.pdf", bbox_inches='tight')
#
####################################
####################################
####################################
####################################
####################################

f = plt.figure()
plt.plot(list(boost_iter_adult_alpha_0_train['param_Boost__n_estimators']), list(boost_iter_adult_alpha_0_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(boost_iter_adult_alpha_0_train['param_Boost__n_estimators']), 
                 list(boost_iter_adult_alpha_0_train['mean_train_score'].values-boost_iter_adult_alpha_0_train['std_train_score']), 
                 list(boost_iter_adult_alpha_0_train['mean_train_score'].values+boost_iter_adult_alpha_0_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(boost_iter_adult_alpha_0_test['param_Boost__n_estimators']), list(boost_iter_adult_alpha_0_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(boost_iter_adult_alpha_0_test['param_Boost__n_estimators']), 
                 list(boost_iter_adult_alpha_0_test['mean_test_score'].values-boost_iter_adult_alpha_0_test['std_test_score']), 
                 list(boost_iter_adult_alpha_0_test['mean_test_score'].values+boost_iter_adult_alpha_0_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Num. of Boost Estimators Complexity -1 vs. Acc.',fontsize=20)
plt.xlabel('Number of Estimators',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

#f.savefig("boost_num_estimators_adult_alpha_0_first_run.pdf", bbox_inches='tight')
#

####################################
####################################
####################################
####################################
####################################

boost_iter_wine = pd.read_csv('output/ITERtestSET_Boost_wine.csv',sep=',')
boost_iter_wine_alpha_0 = pd.read_csv('output/ITERtestSET_Boost_OF_wine.csv',sep=',')

f = plt.figure()
plt.plot(list(boost_iter_wine['param_Boost__n_estimators']), list(boost_iter_wine['train acc']), 
         'k', color='r',label = 'Train')
plt.plot(list(boost_iter_wine['param_Boost__n_estimators']), list(boost_iter_wine['test acc']), 
        'k', color='b', label='Test')
plt.plot(list(boost_iter_wine_alpha_0['param_Boost__n_estimators']), list(boost_iter_wine_alpha_0['train acc']), 
         'k', color='m',label = 'Train Complexity -1')
plt.plot(list(boost_iter_wine_alpha_0['param_Boost__n_estimators']), list(boost_iter_wine_alpha_0['test acc']), 
        'k', color='c', label='Test Complexity -1')
plt.title('Number of Estimators vs. Accuracy',fontsize=20)
plt.xlabel('Number of Estimators',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("boost_estimator_wine_final.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

boost_iter_adult = pd.read_csv('output/ITERtestSET_Boost_adult.csv',sep=',')
boost_iter_adult_alpha_0 = pd.read_csv('output/ITERtestSET_Boost_OF_adult.csv',sep=',')

f = plt.figure()
plt.plot(list(boost_iter_adult['param_Boost__n_estimators']), list(boost_iter_adult['train acc']), 
         'k', color='r',label = 'Train')
plt.plot(list(boost_iter_adult['param_Boost__n_estimators']), list(boost_iter_adult['test acc']), 
        'k', color='b', label='Test')
plt.plot(list(boost_iter_adult_alpha_0['param_Boost__n_estimators']), list(boost_iter_adult_alpha_0['train acc']), 
         'k', color='m',label = 'Train Complexity -1')
plt.plot(list(boost_iter_adult_alpha_0['param_Boost__n_estimators']), list(boost_iter_adult_alpha_0['test acc']), 
        'k', color='c', label='Test Complexity -1')
plt.title('Number of Estimators vs. Accuracy',fontsize=20)
plt.xlabel('Number of Estimators',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("boost_estimator_adult_final.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################
####################################
####################################
####### Decision Trees #############
####################################
####################################
####################################
####################################
####################################
####################################
####################################

dt_wine_reg = pd.read_csv(path + 'output/DT_wine_reg.csv',sep=',')
dt_wine_reg_fit_time = dt_wine_reg[['mean_fit_time',
                                      'param_DT__alpha',
                                      'param_DT__criterion']]

dt_adult_reg = pd.read_csv('output/DT_adult_reg.csv',sep=',')
dt_adult_reg_fit_time = dt_adult_reg[['mean_fit_time',
                                          'param_DT__alpha',
                                          'param_DT__criterion']]

####################################
####################################
####################################
####################################
####################################

estimates = np.unique(dt_wine_reg['param_DT__criterion'])
n_est = len(estimates)
x = dt_wine_reg['param_DT__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = dt_wine_reg['mean_fit_time'].values[i::n_est]

fig, axarr = plt.subplots(1, 2, figsize=(8,6),sharex='col', sharey='row')
alphas = np.unique(dt_wine_reg['param_DT__alpha'])

for i in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(estimates[i],fontsize= 15)
#    axarr[j, i].set_ylim([0,1])
fig.suptitle('Criterion vs. Alpha Time',fontsize= 20)
fig.text(0.04, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.35)
fig.get_figure()
fig
fig.savefig('dt_estimator_alpha_time_wine.pdf')

####################################
####################################
####################################
####################################
####################################

estimates = np.unique(dt_wine_reg['param_DT__criterion'])
n_est = len(estimates)
x = dt_wine_reg['param_DT__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = dt_adult_reg['mean_fit_time'].values[i::n_est]

fig, axarr = plt.subplots(1, 2, figsize=(8,6),sharex='col', sharey='row')
alphas = np.unique(dt_wine_reg['param_DT__alpha'])

for i in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(estimates[i],fontsize= 15)
#    axarr[j, i].set_ylim([0,1])
fig.suptitle('Criterion vs. Alpha Time',fontsize= 20)
fig.text(0.04, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.35)
fig.get_figure()
fig
fig.savefig('dt_estimator_alpha_time_adult.pdf')


####################################
####################################
####################################
####################################
####################################

estimates = np.unique(dt_wine_reg['param_DT__criterion'])
n_est = len(estimates)
x = dt_wine_reg['param_DT__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = dt_wine_reg['mean_train_score'].values[i::n_est]

fig, axarr = plt.subplots(1, 2, figsize=(8,6),sharex='col', sharey='row')
alphas = np.unique(dt_wine_reg['param_DT__alpha'])

for i in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(estimates[i],fontsize= 15)
    axarr[i].set_ylim([0,1])
fig.suptitle('Criterion vs. Alpha Train Accuracy',fontsize= 20)
fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.35)
fig.get_figure()
fig
fig.savefig('dt_criterion_alpha_accuracy_train_wine.pdf')

####################################
####################################
####################################
####################################
####################################

estimates = np.unique(dt_wine_reg['param_DT__criterion'])
n_est = len(estimates)
x = dt_wine_reg['param_DT__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = dt_wine_reg['mean_test_score'].values[i::n_est]

fig, axarr = plt.subplots(1, 2, figsize=(8,6),sharex='col', sharey='row')
alphas = np.unique(dt_wine_reg['param_DT__alpha'])

for i in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(estimates[i],fontsize= 15)
    axarr[i].set_ylim([0,1])
fig.suptitle('Criterion vs. Alpha Test Accuracy',fontsize= 20)
fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.35)
fig.get_figure()
fig
fig.savefig('dt_criterion_alpha_accuracy_test_wine.pdf')

####################################
####################################
####################################
####################################
####################################

estimates = np.unique(dt_adult_reg['param_DT__criterion'])
n_est = len(estimates)
x = dt_adult_reg['param_DT__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = dt_adult_reg['mean_train_score'].values[i::n_est]

fig, axarr = plt.subplots(1, 2, figsize=(8,6),sharex='col', sharey='row')
alphas = np.unique(dt_adult_reg['param_DT__alpha'])

for i in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(estimates[i],fontsize= 15)
    axarr[i].set_ylim([0,1])
fig.suptitle('Criterion vs. Alpha Train Accuracy',fontsize= 20)
fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.35)
fig.get_figure()
fig
fig.savefig('dt_criterion_alpha_accuracy_train_adult.pdf')

####################################
####################################
####################################
####################################
####################################

estimates = np.unique(dt_adult_reg['param_DT__criterion'])
n_est = len(estimates)
x = dt_adult_reg['param_DT__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = dt_adult_reg['mean_test_score'].values[i::n_est]

fig, axarr = plt.subplots(1, 2, figsize=(8,6),sharex='col', sharey='row')
alphas = np.unique(dt_adult_reg['param_DT__alpha'])

for i in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(estimates[i],fontsize= 15)
    axarr[i].set_ylim([0,1])
fig.suptitle('Criterion vs. Alpha Test Accuracy',fontsize= 20)
fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.35)
fig.get_figure()
fig
fig.savefig('dt_criterion_alpha_accuracy_test_adult.pdf')

####################################
####################################
####################################
####################################
####################################

dt_wine_train_LC = pd.read_csv('output/DT_wine_LC_train.csv',sep=',')
dt_wine_test_LC = pd.read_csv('output/DT_wine_LC_test.csv',sep=',')
dt_wine_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
dt_wine_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
dt_wine_train_LC = dt_wine_train_LC.set_index('training_sizes')
dt_wine_test_LC = dt_wine_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(dt_wine_train_LC,axis=1)
mean_test_size_score = np.mean(dt_wine_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("dt_wine_train_LC.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

dt_adult_train_LC = pd.read_csv('output/DT_adult_LC_train.csv',sep=',')
dt_adult_test_LC = pd.read_csv('output/DT_adult_LC_test.csv',sep=',')
dt_adult_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
dt_adult_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
dt_adult_train_LC = dt_adult_train_LC.set_index('training_sizes')
dt_adult_test_LC = dt_adult_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(dt_adult_train_LC,axis=1)
mean_test_size_score = np.mean(dt_adult_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Adult Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("dt_adult_train_LC.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

dt_wine_nodes = pd.read_csv('output/DT_wine_nodecounts.csv',sep=',')
dt_wine_nodes.columns = ['alpha', 'nodes']

fig, axarr = plt.subplots(1, 1, figsize=(8,6),sharex='col', sharey='row')


df = dt_wine_nodes.set_index('alpha')
df['nodes'].sort_index().plot.bar(ax=axarr)
df
#    axarr[j, i].set_ylim([0,1])
fig.suptitle('Alpha vs. Number of Nodes in Tree',fontsize= 20)
fig.text(0.04, 0.5, 'Number of Nodes in Tree', va='center', rotation='vertical',fontsize= 15)
fig
#fig.savefig("dt_num_nodes_wine.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

dt_adult_nodes = pd.read_csv('output/DT_adult_nodecounts.csv',sep=',')
dt_adult_nodes.columns = ['alpha', 'nodes']

fig, axarr = plt.subplots(1, 1, figsize=(8,6),sharex='col', sharey='row')


df = dt_adult_nodes.set_index('alpha')
df['nodes'].sort_index().plot.bar(ax=axarr)
df
#    axarr[j, i].set_ylim([0,1])
fig.suptitle('Alpha vs. Number of Nodes in Tree',fontsize= 20)
fig.text(0.04, 0.5, 'Number of Nodes in Tree', va='center', rotation='vertical',fontsize= 15)
fig
#fig.savefig("dt_num_nodes_adult.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################
####################################
####################################
############## SVM #################
####################################
####################################
####################################
####################################
####################################
####################################
####################################

SVM_wine_reg = pd.read_csv(path + 'output/SVM_RBF_wine_reg.csv',sep=',')
SVM_wine_reg_fit_time = SVM_wine_reg[['mean_fit_time',
                                      'param_SVM__alpha',
                                      'param_SVM__gamma_frac']]

SVM_adult_reg = pd.read_csv(path + 'output/SVM_RBF_small_adult_reg.csv',sep=',')
SVM_adult_reg_fit_time = SVM_adult_reg[['mean_fit_time',
                                          'param_SVM__alpha',
                                          'param_SVM__gamma_frac']]

####################################
####################################
####################################
####################################
####################################

gamma = np.unique(SVM_wine_reg['param_SVM__gamma_frac'])
n_est = len(gamma)
x = SVM_wine_reg['param_SVM__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = SVM_wine_reg['mean_fit_time'].values[i::n_est]

gamma = np.unique(SVM_wine_reg['param_SVM__gamma_frac'])
fig, axarr = plt.subplots(2, 2, figsize=(6,6),sharex='col', sharey='row')
alphas = np.unique(SVM_wine_reg['param_SVM__alpha'])

for i in range(2):
  for j in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(gamma[i+2*j],fontsize= 15)
#    axarr[j, i].set_ylim([0,1])
fig.suptitle('Gamma vs. Alpha Time',fontsize= 20)
fig.text(0.04, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.15)
fig.get_figure()
fig
fig.savefig('SVM_gamma_alpha_time_wine.pdf')

####################################
####################################
####################################
####################################
####################################

gamma = np.unique(SVM_adult_reg['param_SVM__gamma_frac'])
n_est = len(gamma)
x = SVM_adult_reg['param_SVM__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = SVM_adult_reg['mean_fit_time'].values[i::n_est]

gamma = np.unique(SVM_adult_reg['param_SVM__gamma_frac'])
fig, axarr = plt.subplots(2, 2, figsize=(6,6),sharex='col', sharey='row')
alphas = np.unique(SVM_adult_reg['param_SVM__alpha'])

for i in range(2):
  for j in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(gamma[i+2*j],fontsize= 15)
#    axarr[j, i].set_ylim([0,1])
fig.suptitle('Gamma vs. Alpha Time',fontsize= 20)
fig.text(0.04, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.15)
fig.get_figure()
fig
fig.savefig('SVM_gamma_alpha_time_adult.pdf')


####################################
####################################
####################################
####################################
####################################

gamma = np.unique(SVM_wine_reg['param_SVM__gamma_frac'])
n_est = len(gamma)
x = SVM_wine_reg['param_SVM__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = SVM_wine_reg['mean_train_score'].values[i::n_est]

fig, axarr = plt.subplots(2, 2, figsize=(6,6),sharex='col', sharey='row')
alphas = np.unique(SVM_wine_reg['param_SVM__alpha'])

for i in range(2):
  for j in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(gamma[i+2*j],fontsize= 15)
    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Alpha Train Accuracy',fontsize= 20)
fig.text(0.03, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.15)
fig.get_figure()
fig
fig.savefig('SVM_gamma_alpha_accuracy_train_wine.pdf')

####################################
####################################
####################################
####################################
####################################

gamma = np.unique(SVM_wine_reg['param_SVM__gamma_frac'])
n_est = len(gamma)
x = SVM_wine_reg['param_SVM__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = SVM_wine_reg['mean_test_score'].values[i::n_est]

fig, axarr = plt.subplots(2, 2, figsize=(6,6),sharex='col', sharey='row')
alphas = np.unique(SVM_wine_reg['param_SVM__alpha'])

for i in range(2):
  for j in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(gamma[i+2*j],fontsize= 15)
    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Alpha Test Accuracy',fontsize= 20)
fig.text(0.03, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.15)
fig.get_figure()
fig
fig.savefig('SVM_gamma_alpha_accuracy_test_wine.pdf')
#
####################################
####################################
####################################
####################################
####################################

gamma = np.unique(SVM_adult_reg['param_SVM__gamma_frac'])
n_est = len(gamma)
x = SVM_adult_reg['param_SVM__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = SVM_adult_reg['mean_train_score'].values[i::n_est]

fig, axarr = plt.subplots(2, 2, figsize=(6,6),sharex='col', sharey='row')
alphas = np.unique(SVM_adult_reg['param_SVM__alpha'])

for i in range(2):
  for j in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(gamma[i+2*j],fontsize= 15)
    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Alpha Train Accuracy',fontsize= 20)
fig.text(0.03, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.15)
fig.get_figure()
fig
fig.savefig('SVM_gamma_alpha_accuracy_train_adult.pdf')

####################################
####################################
####################################
####################################
####################################

gamma = np.unique(SVM_adult_reg['param_SVM__gamma_frac'])
n_est = len(gamma)
x = SVM_adult_reg['param_SVM__alpha']
y = np.zeros(n_est)
y = list(y)
for i in range(n_est):
  y[i] = SVM_adult_reg['mean_test_score'].values[i::n_est]

fig, axarr = plt.subplots(2, 2, figsize=(6,6),sharex='col', sharey='row')
alphas = np.unique(SVM_adult_reg['param_SVM__alpha'])

for i in range(2):
  for j in range(2):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+2*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot.bar(ax=axarr[j][i])
    axarr[j, i].set_title(gamma[i+2*j],fontsize= 15)
    axarr[j, i].set_ylim([0,1])
fig.suptitle('Estimator vs. Alpha Test Accuracy',fontsize= 20)
fig.text(0.03, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.15)
fig.get_figure()
fig
fig.savefig('SVM_gamma_alpha_accuracy_test_adult.pdf')

####################################
####################################
####################################
####################################
####################################

SVM_wine_train_LC = pd.read_csv('output/SVM_RBF_wine_LC_train.csv',sep=',')
SVM_wine_test_LC = pd.read_csv('output/SVM_RBF_wine_LC_test.csv',sep=',')
SVM_wine_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
SVM_wine_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
SVM_wine_train_LC = SVM_wine_train_LC.set_index('training_sizes')
SVM_wine_test_LC = SVM_wine_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(SVM_wine_train_LC,axis=1)
mean_test_size_score = np.mean(SVM_wine_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("SVM_wine_train_LC.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

SVM_adult_train_LC = pd.read_csv('output/SVM_RBF_small_adult_LC_train.csv',sep=',')
SVM_adult_test_LC = pd.read_csv('output/SVM_RBF_small_adult_LC_test.csv',sep=',')
SVM_adult_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
SVM_adult_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
SVM_adult_train_LC = SVM_adult_train_LC.set_index('training_sizes')
SVM_adult_test_LC = SVM_adult_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(SVM_adult_train_LC,axis=1)
mean_test_size_score = np.mean(SVM_adult_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("SVM_adult_train_LC.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

SVM_iter_wine = pd.read_csv(path + 'output/ITER_base_SVM_RBF_wine.csv',sep=',')
SVM_iter_wine_alpha_0 = pd.read_csv(path + 'output/ITER_base_SVM_RBF_OF_wine.csv',sep=',')
SVM_iter_wine_train = SVM_iter_wine[['mean_train_score','param_SVM__n_iter','std_train_score']].copy()
SVM_iter_wine_test = SVM_iter_wine[['mean_test_score','param_SVM__n_iter','std_test_score']].copy()
SVM_iter_wine_alpha_0_train = SVM_iter_wine_alpha_0[['mean_train_score','param_SVM__n_iter','std_train_score']].copy()
SVM_iter_wine_alpha_0_test = SVM_iter_wine_alpha_0[['mean_test_score','param_SVM__n_iter','std_test_score']].copy()

f = plt.figure()
plt.plot(list(SVM_iter_wine_train['param_SVM__n_iter']), list(SVM_iter_wine_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(SVM_iter_wine_train['param_SVM__n_iter']), 
                 list(SVM_iter_wine_train['mean_train_score'].values-SVM_iter_wine_train['std_train_score']), 
                 list(SVM_iter_wine_train['mean_train_score'].values+SVM_iter_wine_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(SVM_iter_wine_test['param_SVM__n_iter']), list(SVM_iter_wine_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(SVM_iter_wine_test['param_SVM__n_iter']), 
                 list(SVM_iter_wine_test['mean_test_score'].values-SVM_iter_wine_test['std_test_score']), 
                 list(SVM_iter_wine_test['mean_test_score'].values+SVM_iter_wine_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Max Number of SVM Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Max Number of Iterations ',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("SVM_num_estimators_wine_first_run.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################

f = plt.figure()
plt.plot(list(SVM_iter_wine_alpha_0_train['param_SVM__n_iter']), list(SVM_iter_wine_alpha_0_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(SVM_iter_wine_alpha_0_train['param_SVM__n_iter']), 
                 list(SVM_iter_wine_alpha_0_train['mean_train_score'].values-SVM_iter_wine_alpha_0_train['std_train_score']), 
                 list(SVM_iter_wine_alpha_0_train['mean_train_score'].values+SVM_iter_wine_alpha_0_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(SVM_iter_wine_alpha_0_test['param_SVM__n_iter']), list(SVM_iter_wine_alpha_0_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(SVM_iter_wine_alpha_0_test['param_SVM__n_iter']), 
                 list(SVM_iter_wine_alpha_0_test['mean_test_score'].values-SVM_iter_wine_alpha_0_test['std_test_score']), 
                 list(SVM_iter_wine_alpha_0_test['mean_test_score'].values+SVM_iter_wine_alpha_0_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Max Number of SVM Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Max Number of Iterations ',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("SVM_num_estimators_wine_alpha_0_first_run.pdf", bbox_inches='tight')
#
####################################
####################################
####################################
####################################
####################################

SVM_iter_adult = pd.read_csv(path+'output/ITER_base_SVM_RBF_adult.csv',sep=',')
SVM_iter_adult_alpha_0 = pd.read_csv(path+'output/ITER_base_SVM_RBF_OF_adult.csv',sep=',')
SVM_iter_adult_train = SVM_iter_adult[['mean_train_score','param_SVM__n_iter','std_train_score']].copy()
SVM_iter_adult_test = SVM_iter_adult[['mean_test_score','param_SVM__n_iter','std_test_score']].copy()
SVM_iter_adult_alpha_0_train = SVM_iter_adult_alpha_0[['mean_train_score','param_SVM__n_iter','std_train_score']].copy()
SVM_iter_adult_alpha_0_test = SVM_iter_adult_alpha_0[['mean_test_score','param_SVM__n_iter','std_test_score']].copy()

f = plt.figure()
plt.plot(list(SVM_iter_adult_train['param_SVM__n_iter']), list(SVM_iter_adult_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(SVM_iter_adult_train['param_SVM__n_iter']), 
                 list(SVM_iter_adult_train['mean_train_score'].values-SVM_iter_adult_train['std_train_score']), 
                 list(SVM_iter_adult_train['mean_train_score'].values+SVM_iter_adult_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(SVM_iter_adult_test['param_SVM__n_iter']), list(SVM_iter_adult_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(SVM_iter_adult_test['param_SVM__n_iter']), 
                 list(SVM_iter_adult_test['mean_test_score'].values-SVM_iter_adult_test['std_test_score']), 
                 list(SVM_iter_adult_test['mean_test_score'].values+SVM_iter_adult_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Max Number of SVM Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Max Number of Iterations ',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("SVM_num_estimators_adult_first_run.pdf", bbox_inches='tight')
#
####################################
####################################
####################################
####################################
####################################

f = plt.figure()
plt.plot(list(SVM_iter_adult_alpha_0_train['param_SVM__n_iter']), list(SVM_iter_adult_alpha_0_train['mean_train_score']), 
         'k', color='#3F7F4C',label = 'Train')
plt.fill_between(list(SVM_iter_adult_alpha_0_train['param_SVM__n_iter']), 
                 list(SVM_iter_adult_alpha_0_train['mean_train_score'].values-SVM_iter_adult_alpha_0_train['std_train_score']), 
                 list(SVM_iter_adult_alpha_0_train['mean_train_score'].values+SVM_iter_adult_alpha_0_train['std_train_score']),
    alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99', antialiased=True)
plt.plot(list(SVM_iter_adult_alpha_0_test['param_SVM__n_iter']), list(SVM_iter_adult_alpha_0_test['mean_test_score']), 
        'k', color='#CC4F1B', label='Test')
plt.fill_between(list(SVM_iter_adult_alpha_0_test['param_SVM__n_iter']), 
                 list(SVM_iter_adult_alpha_0_test['mean_test_score'].values-SVM_iter_adult_alpha_0_test['std_test_score']), 
                 list(SVM_iter_adult_alpha_0_test['mean_test_score'].values+SVM_iter_adult_alpha_0_test['std_test_score']),
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Max Number of SVM Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Max Number of Iterations ',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("SVM_num_estimators_adult_alpha_0_first_run.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################

SVM_iter_wine = pd.read_csv(path + 'output/ITERtestSET_SVM_RBF_wine.csv',sep=',')
SVM_iter_wine_alpha_0 = pd.read_csv(path + 'output/ITERtestSET_SVM_RBF_OF_wine.csv',sep=',')
SVM_iter_wine['train acc']
SVM_iter_wine_alpha_0['train acc']
f = plt.figure()
plt.plot(list(SVM_iter_wine['param_SVM__n_iter']), list(SVM_iter_wine['train acc']), 
         'k', color='r',label = 'Train')
plt.plot(list(SVM_iter_wine['param_SVM__n_iter']), list(SVM_iter_wine['test acc']), 
        'k', color='b', label='Test')
plt.plot(list(SVM_iter_wine_alpha_0['param_SVM__n_iter']), list(SVM_iter_wine_alpha_0['train acc']), 
         'k', color='m',label = 'Train Alpha 0')
plt.plot(list(SVM_iter_wine_alpha_0['param_SVM__n_iter']), list(SVM_iter_wine_alpha_0['test acc']), 
        'k', color='c', label='Test Alpha 0')
plt.title('Max Number of Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("SVM_estimator_wine_final.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

SVM_iter_adult = pd.read_csv('output/ITERtestSET_SVM_RBF_adult.csv',sep=',')
SVM_iter_adult_alpha_0 = pd.read_csv('output/ITERtestSET_SVM_RBF_OF_adult.csv',sep=',')

f = plt.figure()
plt.plot(list(SVM_iter_adult['param_SVM__n_iter']), list(SVM_iter_adult['train acc']), 
         'k', color='r',label = 'Train')
plt.plot(list(SVM_iter_adult['param_SVM__n_iter']), list(SVM_iter_adult['test acc']), 
        'k', color='b', label='Test')
plt.plot(list(SVM_iter_adult_alpha_0['param_SVM__n_iter']), list(SVM_iter_adult_alpha_0['train acc']), 
         'k', color='m',label = 'Train Alpha 0')
plt.plot(list(SVM_iter_adult_alpha_0['param_SVM__n_iter']), list(SVM_iter_adult_alpha_0['test acc']), 
        'k', color='c', label='Test Alpha 0')
plt.title('Max Number of Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

f.savefig("SVM_estimator_adult_final.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################
####################################
########### KNN ####################
####################################
####################################
####################################
####################################
####################################
####################################
####################################
####################################

KNN_wine_reg = pd.read_csv(path + 'output/KNN_wine_reg.csv',sep=',')
KNN_wine_reg_fit_time = KNN_wine_reg[['mean_fit_time','param_KNN__metric','param_KNN__n_neighbors','param_KNN__weights']]

KNN_adult_reg = pd.read_csv('output/KNN_adult_reg.csv',sep=',')
KNN_adult_reg_fit_time = KNN_adult_reg[['mean_fit_time','param_KNN__metric','param_KNN__n_neighbors','param_KNN__weights']]


#######################################
#######################################
#######################################
#######################################
#######################################

KNN_relu_reg = KNN_wine_reg[KNN_wine_reg['param_KNN__weights']==\
                            'uniform'][['mean_fit_time',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_wine_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_relu_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_relu_reg['mean_fit_time'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_wine_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
#    axarr[i].set_ylim([0,1])
fig.suptitle('Uniform Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_wine_uniform_time.pdf')

#######################################
#######################################
#######################################
#######################################
#######################################

KNN_relu_reg = KNN_adult_reg[KNN_adult_reg['param_KNN__weights']==\
                            'uniform'][['mean_fit_time',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_adult_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_relu_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_relu_reg['mean_fit_time'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_adult_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
#    axarr[i].set_ylim([0,1])
fig.suptitle('Uniform Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_adult_uniform_time.pdf')

#######################################
#######################################
#######################################
#######################################
#######################################

KNN_dist_reg = KNN_wine_reg[KNN_wine_reg['param_KNN__weights']==\
                            'distance'][['mean_fit_time',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_wine_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_fit_time'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_wine_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
#    axarr[i].set_ylim([0,1])
fig.suptitle('Distance Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_wine_distance_time.pdf')

#######################################
#######################################
#######################################
#######################################
#######################################

KNN_dist_reg = KNN_adult_reg[KNN_adult_reg['param_KNN__weights']==\
                            'distance'][['mean_fit_time',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_adult_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_fit_time'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_adult_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
#    axarr[i].set_ylim([0,1])
fig.suptitle('Distance Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Time (Seconds)', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_adult_distance_time.pdf')


####################################
####################################
####################################
####################################
####################################

KNN_dist_reg = KNN_wine_reg[KNN_wine_reg['param_KNN__weights']==\
                            'uniform'][['mean_train_score',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_wine_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_train_score'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_wine_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
    axarr[i].set_ylim([0,1])
fig.suptitle('Unifrom Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_wine_uniform_score_train.pdf')

####################################
####################################
####################################
####################################
####################################

KNN_dist_reg = KNN_wine_reg[KNN_wine_reg['param_KNN__weights']==\
                            'distance'][['mean_train_score',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_wine_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_train_score'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_wine_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
    axarr[i].set_ylim([0,1])
fig.suptitle('Distance Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_wine_distance_score_train.pdf')

####################################
####################################
####################################
####################################
####################################

KNN_dist_reg = KNN_adult_reg[KNN_adult_reg['param_KNN__weights']==\
                            'uniform'][['mean_train_score',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_adult_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_train_score'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_adult_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
    axarr[i].set_ylim([0,1])
fig.suptitle('Unifrom Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_adult_uniform_score_train.pdf')

####################################
####################################
####################################
####################################
####################################

KNN_dist_reg = KNN_adult_reg[KNN_adult_reg['param_KNN__weights']==\
                            'distance'][['mean_train_score',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_adult_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_train_score'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_adult_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
    axarr[i].set_ylim([0,1])
fig.suptitle('Distance Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_adult_distance_score_train.pdf')

####################################
####################################
####################################
####################################
####################################

KNN_dist_reg = KNN_wine_reg[KNN_wine_reg['param_KNN__weights']==\
                            'uniform'][['mean_test_score',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_wine_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_test_score'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_wine_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
    axarr[i].set_ylim([0,1])
fig.suptitle('Unifrom Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_wine_uniform_score_test.pdf')

####################################
####################################
####################################
####################################
####################################

KNN_dist_reg = KNN_wine_reg[KNN_wine_reg['param_KNN__weights']==\
                            'distance'][['mean_test_score',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_wine_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_test_score'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_wine_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
    axarr[i].set_ylim([0,1])
fig.suptitle('Distance Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_wine_distance_score_test.pdf')

####################################
####################################
####################################
####################################
####################################

KNN_dist_reg = KNN_adult_reg[KNN_adult_reg['param_KNN__weights']==\
                            'uniform'][['mean_test_score',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_adult_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_test_score'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_adult_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
    axarr[i].set_ylim([0,1])
fig.suptitle('Unifrom Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_adult_uniform_score_test.pdf')

####################################
####################################
####################################
####################################
####################################

KNN_dist_reg = KNN_adult_reg[KNN_adult_reg['param_KNN__weights']==\
                            'distance'][['mean_test_score',
                                     'param_KNN__metric',
                                     'param_KNN__n_neighbors']]
metric_names = np.unique(KNN_adult_reg_fit_time['param_KNN__metric'])
n_met = len(metric_names)
x = KNN_dist_reg['param_KNN__n_neighbors']
y = np.zeros(n_met)
y = list(y)
for i in range(n_met):
  y[i] = KNN_dist_reg['mean_test_score'].values[i::n_met]

fig, axarr = plt.subplots(3, 1, figsize=(6, 12),sharex='col', sharey='row')
nbrs = np.unique(KNN_adult_reg_fit_time['param_KNN__n_neighbors'])

for i in range(3):
    df = pd.concat([pd.DataFrame(nbrs),pd.DataFrame(y[i])],axis=1)
    df.columns = ['Neighbors','alphas']
    df = df.set_index('Neighbors')
    df['alphas'].sort_index().plot.bar(ax=axarr[i])
    axarr[i].set_title(metric_names[i],fontsize= 20)
    axarr[i].set_ylim([0,1])
fig.suptitle('Distance Weights',fontsize= 20)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(left = 0.15)
fig.get_figure()
fig
fig.savefig('KNN_metric_nbrs_adult_distance_score_test.pdf')

####################################
####################################
####################################
####################################
####################################

KNN_wine_train_LC = pd.read_csv('output/KNN_wine_LC_train.csv',sep=',')
KNN_wine_test_LC = pd.read_csv('output/KNN_wine_LC_test.csv',sep=',')
KNN_wine_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
KNN_wine_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
KNN_wine_train_LC = KNN_wine_train_LC.set_index('training_sizes')
KNN_wine_test_LC = KNN_wine_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(KNN_wine_train_LC,axis=1)
mean_test_size_score = np.mean(KNN_wine_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("KNN_wine_train_LC.pdf", bbox_inches='tight')


####################################
####################################
####################################
####################################
####################################

KNN_adult_train_LC = pd.read_csv('output/KNN_adult_LC_train.csv',sep=',')
KNN_adult_test_LC = pd.read_csv('output/KNN_adult_LC_test.csv',sep=',')
KNN_adult_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
KNN_adult_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
KNN_adult_train_LC = KNN_adult_train_LC.set_index('training_sizes')
KNN_adult_test_LC = KNN_adult_test_LC.set_index('training_sizes')

mean_train_size_score = np.mean(KNN_adult_train_LC,axis=1)
mean_test_size_score = np.mean(KNN_adult_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Adult Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()

#f.savefig("KNN_adult_train_LC.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################

KNN_iter_adult = pd.read_csv('output/KNN_adult_timing.csv',sep=',')
KNN_iter_adult_alpha_0 = pd.read_csv('output/ITERtestSET_KNN_adult.csv',sep=',')

f = plt.figure()
plt.plot(list(KNN_iter_adult['param_MLP__max_iter']), list(KNN_iter_adult['train acc']), 
         'k', color='r',label = 'Train')
plt.plot(list(KNN_iter_adult['param_MLP__max_iter']), list(KNN_iter_adult['test acc']), 
        'k', color='b', label='Test')
plt.plot(list(KNN_iter_adult_alpha_0['param_MLP__max_iter']), list(KNN_iter_adult_alpha_0['train acc']), 
         'k', color='m',label = 'Train Alpha 0')
plt.plot(list(KNN_iter_adult_alpha_0['param_MLP__max_iter']), list(KNN_iter_adult_alpha_0['test acc']), 
        'k', color='c', label='Test Alpha 0')
plt.title('Max Number of Iterations vs. Accuracy',fontsize=20)
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)

#f.savefig("KNN_max_iter_adult_final.pdf", bbox_inches='tight')

####################################
####################################
####################################
####################################
####################################



####################################
####################################
####################################
####################################
####################################





#fig.savefig('KNN_reg_alpha_score.pdf')


# sns.heatmap(vals.corr(),annot=True, fmt='.2f') #Use heat map to show little colinearity.
# g = sns.pairplot(vals, hue="income", palette="husl") #send to figure.

#
##histogram plot for frequencies of the ratings
#out = pd.cut(wine['quality'], bins=range(2,10), include_lowest=True)
#ax = out.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(7,5))
#ax.set_title('Wine Quality Frequency')
#ax.set_xlabel('Rating')
#ax.set_ylabel('Frequency')
#ax.set_xticklabels([3,4,5,6,7,8,9])
#rects = ax.patches
#count,_ = np.histogram(wine['quality'], bins = range(3,11))
#
## Make some labels.
#labels = list(count)
#
#for rect, label in zip(rects, labels):
#    height = rect.get_height()
#    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
#            ha='center', va='bottom')
#
#
#mad_score_wine = robust.mad(wine['quality'])
#mad_score_wine 
#wine[abs(wine['quality'].values-stats.median(wine['quality'].values))/mad_score_wine > 2]
#
#mad_score_adult = robust.mad(vals['income'])
#vals[abs(vals['income'].values-stats.median(vals['income'].values))/mad_score_adult > 2]
#
#wineX = wine.drop('quality',1).copy().values
#wineY = wine['quality'].copy()
#
#wineY[wineY.isin([1,2,3,4,5,6])] = 1
#wineY[wineY.isin([7,8,9,10])] = 0
#
#adult_trnX, adult_tstX, adult_trnY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
#wine_trnX, wine_tstX, wine_trnY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)
#
##wine.to_csv('full_wine_data.csv')
#
#sns.heatmap(wine.corr(),annot=True, fmt='.2f') #Use heat map to show little colinearity.
#g = sns.pairplot(wine, hue="quality", palette="husl") #send to figure.
#
#sns.heatmap(vals.corr(),annot=True, fmt='.2f') #Use heat map to show little colinearity.
#g = sns.pairplot(vals, hue="income", palette="husl") #send to figure.
#
#
##summary table
#wine.describe().to_latex()
#wine.describe()
#
#len(wineY[wineY==1])/len(wineY)
#adult_df.describe().to_latex()
#adult_df.describe()
#adult_df.describe(include='O').to_latex()
#adult_df.describe(include='O')
#np.unique(adult_df['income'].values)
#adult_df['income'].value_counts().plot(kind='bar')
##There are no missing data.
#wine.isnull().values.any()
#adult_df.isnull().values.any()
#
#wine.info()
#adult_df.info()
#
#
#
