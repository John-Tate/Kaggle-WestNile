import pandas as pd
from keras.models import Sequential
import theano
import numpy as np

from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

from sklearn.decomposition import PCA


train = pd.read_csv('trainComb.csv')
test = pd.read_csv('testComb.csv')

st1cols = [u'Latitude', u'Longitude',u'Week', u'Month', u'Tmax_x',
       u'Tmin_x', u'Tavg_x', u'DewPoint_x', u'WetBulb_x',
     u'StnPressure_x',]

X = train[st1cols]
y = train['WnvPresent']

X = np.array(X)
y = np.array(y)
yc = np_utils.to_categorical(y)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

Xt = test[st1cols]
Xt = np.array(Xt)
scaler = StandardScaler()
scaler.fit(Xt)
Xt = scaler.transform(Xt)

###############
##############

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.0035,loss='deviance',max_features=8,max_depth=7,subsample=1)




clf = gb
clf.fit(X, y)

gb_pred = clf.predict_proba(Xt)

len(gb_pred[:,1])

pred_test = test.copy()


pred_test['pred_prob'] = gb_pred[:,1]

pred_test.head()

pred_test_fin = pd.concat([pred_test['pred_prob'],pred_test[st1cols],pred_test['Trap']],axis=1)

pred_test_fin.head()

gb_pred_grouped_trap = pred_test_fin.groupby(['Trap']).mean()

gb_pred_grouped_week = pred_test_fin.groupby(['Week']).mean()



gb_pred_grouped_week.head()
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib

gb_pred_grouped_week.index

sns.barplot(gb_pred_grouped_week.index,gb_pred_grouped_week['pred_prob'],palette='inferno')


submission = pd.read_csv('../assets/sampleSubmission.csv')
subdat = pd.DataFrame(submission)



len(submission)
len(gb_pred)

subdat['WnvPresent'] = gb_pred[:,1]

subdat

subdat.to_csv('test_sub_gb4.csv', index=False)

from sklearn import metrics
import pandas as pd
from ggplot import *

preds = clf.predict_proba(Xtest)[:,1]
fpr, tpr, _ = metrics.roc_curve(ytest, preds)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')
