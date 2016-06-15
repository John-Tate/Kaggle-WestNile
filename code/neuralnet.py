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


train = pd.read_csv('../assets/trainComb.csv')

feature_cols =  ['Species','NumMosquitos','Week','Station','Tmax','Tmin','Tavg','DewPoint',
                'SeaLevel','ResultSpeed','ResultDir','AvgSpeed']

X = train[feature_cols]
y = train['WnvPresent']

X = np.array(X)
X.shape

y = np.array(y)
y.shape


y = np_utils.to_categorical(y)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)

X_train.shape

input_dim = X_train.shape[1]
output_dim = 2

# model = Sequential()
#
# model.add(Dense(16,input_dim=X_train.shape[1]))
# model.add(Activation('relu'))
#
# model.add(Dense(output_dim=1))
# model.add(Activation("softmax"))
#
# model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

model = Sequential()
model.add(Dense(32, input_dim=input_dim))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(32))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(output_dim))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer="sgd")

model.fit(X_train, y_train, nb_epoch=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

y_probs = model.predict_proba(X_test, verbose=0)
y_preds = model.predict(X_test)

y_test

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_probs)
rf_roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

roc = metrics.roc_auc_score(y_test, y_probs)
print("AUC:", roc)
