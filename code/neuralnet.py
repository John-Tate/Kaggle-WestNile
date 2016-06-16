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

def annModel(inputD,outputD):
    model = Sequential()

    model.add(Dense(32, input_dim=inputD))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(outputD))
    model.add(Activation('softmax'))

    model.compile(loss='mean_squared_error', optimizer="adadelta")
    return model


train = pd.read_csv('../assets/trainComb.csv')

feature_cols =  ['Species','NumMosquitos','Week','Station','Tmax','Tmin','Tavg','DewPoint',
                'SeaLevel','ResultSpeed','ResultDir','AvgSpeed']

X = train[feature_cols]
y = train['WnvPresent']

X = np.array(X)
X.shape

y = np.array(y)
y.shape


yc = np_utils.to_categorical(y)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)


from sklearn.cross_validation import KFold
kf = KFold(len(y), n_folds=4 ,shuffle=True, random_state=0)

inputD = X_train.shape[1]
outputD = 2

inputD


auc_scores = []
acc_scores = []

for training, testing in kf:
    print "Fold"
    X_train = X[training]
    X_test = X[testing]
    y_train = yc[training]
    y_test = yc[testing]
    y_true = y[testing]

    model = annModel(inputD,outputD)
    model.fit(X_train, y_train, nb_epoch=100, batch_size=16, validation_data=(X_test, y_test), verbose=0)

    y_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    roc = metrics.roc_auc_score(y_test, y_probs)
    auc_scores.append(roc)

print np.mean(auc_scores)

print auc_scores.round(4)

for a in auc_scores:
    print "%.4f" % a
