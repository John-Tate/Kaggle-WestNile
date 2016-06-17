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


def annModel(inputD,outputD):
    model = Sequential()

    model.add(Dense(32, input_dim=inputD))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(outputD))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adadelta")
    return model


train = pd.read_csv('trainComb.csv')


train.columns

feature_cols =  ['Species','Week','Station','Tmax','Tmin','Tavg','DewPoint',
                'SeaLevel','ResultSpeed','ResultDir','AvgSpeed']

X = train.drop(['WnvPresent','NumMosquitos'],axis=1)
y = train['WnvPresent']

from sklearn.decomposition import PCA


pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)


X = np.array(X_pca)
y = np.array(y)
yc = np_utils.to_categorical(y)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)


kf = KFold(len(y), n_folds=5 ,shuffle=True, random_state=0)
inputD = X.shape[1]
outputD = 2


auc_scores = []
# acc_scores = []
fold = 0

for training, testing in kf:
    fold += 1
    print "Fold Start", fold
    X_train = X[training]
    X_test = X[testing]
    y_train = yc[training]
    y_test = yc[testing]
    y_true = y[testing]

    model = annModel(inputD,outputD)
    model.fit(X_train, y_train, nb_epoch=100, batch_size=16, validation_data=(X_test, y_test), verbose=0)

    y_probs = model.predict_proba(X_test,verbose=0)
    y_pred = model.predict(X_test)

    roc = metrics.roc_auc_score(y_test, y_probs)
    auc_scores.append(roc)
    print "Fold Complete", fold


print np.mean(auc_scores)

for a in auc_scores:
    print "%.4f" % a

#### Train on all the data


inputD = X.shape[1]
outputD = 2
model = annModel(inputD,outputD)
model.fit(X, yc, nb_epoch=200, batch_size=16, verbose=0)

#### Run it on the test data

test_data = pd.read_csv('../assets/testComb.csv')



Xt = test_data[feature_cols]

Xt = np.array(Xt)


scaler = StandardScaler()
scaler.fit(Xt)
Xt = scaler.transform(Xt)

sub_probs = model.predict_proba(Xt,verbose=0)

submission = pd.read_csv('../assets/sampleSubmission.csv')

subdat = pd.DataFrame(submission)

subdat.head()

len(submission)
len(sub_probs)

subdat['WnvPresent'] = sub_probs[:,1]

subdat.to_csv('test_sub.csv', index=False)

subdat.head()
