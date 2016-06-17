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


def annModel(inputD,outputD):
    model = Sequential()

    model.add(Dense(32, input_dim=inputD))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(outputD))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adadelta")
    return model


train = pd.read_csv('trainComb.csv')


train.columns

st1cols = [ u'Species', u'Latitude', u'Longitude',u'Week', u'Month', u'Tmax_x',
       u'Tmin_x', u'Tavg_x', u'DewPoint_x', u'WetBulb_x',
     u'StnPressure_x',]
len(st1cols)

return [month, week, latitude, longitude, tmax, tmin, tavg, dewpoint, wetbulb, pressure]


PCA_cols = [u'Tmax_x',u'Tmin_x', u'Tavg_x', u'Depart_x', u'DewPoint_x', u'WetBulb_x',
u'Heat_x', u'Cool_x', u'PrecipTotal_x', u'StnPressure_x', u'SeaLevel_x',
u'ResultSpeed_x', u'ResultDir_x', u'AvgSpeed_x', u'Tmax_y', u'Tmin_y',
u'Tavg_y', u'DewPoint_y', u'WetBulb_y', u'Heat_y', u'Cool_y',
u'PrecipTotal_y', u'StnPressure_y', u'SeaLevel_y', u'ResultSpeed_y',
u'ResultDir_y', u'AvgSpeed_y']

feature_cols = [u'Species', u'Trap', u'Latitude', u'Longitude', u'AddressAccuracy',
u'Week', u'Month', u'Day', u'Tmax_x',
u'Tmin_x', u'Tavg_x', u'Depart_x', u'DewPoint_x', u'WetBulb_x',
u'Heat_x', u'Cool_x', u'PrecipTotal_x', u'StnPressure_x', u'SeaLevel_x',
u'ResultSpeed_x', u'ResultDir_x', u'AvgSpeed_x', u'Tmax_y', u'Tmin_y',
u'Tavg_y', u'DewPoint_y', u'WetBulb_y', u'Heat_y', u'Cool_y',
u'PrecipTotal_y', u'StnPressure_y', u'SeaLevel_y', u'ResultSpeed_y',
u'ResultDir_y', u'AvgSpeed_y']

train_weather = train[PCA_cols]
train2 = train.drop(PCA_cols,axis=1)


# pca = PCA(n_components=10)
# train_weather_pca = pca.fit_transform(train_weather)
#
# train2.shape
# train_weather_pca.shape
#
# train_weather_pca = pd.DataFrame(train_weather_pca)
# train_weather_pca.head()
# train2.head()
#
# train = pd.concat([train2,train_weather_pca],axis=1)
#
# len(train.columns)
# train.head()
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

kf = KFold(len(y), n_folds=4 ,shuffle=True, random_state=0)
inputD = X.shape[1]
outputD = 2


auc_scores = []
# acc_scores = []
fold = 0

annScores = []

for training, testing in kf:
    fold += 1
    print "Fold Start", fold
    X_train = X[training]
    X_test = X[testing]
    y_train = yc[training]
    y_test = yc[testing]
    y_true = y[testing]

    model = annModel(inputD,outputD)
    model.fit(X_train, y_train, nb_epoch=50, batch_size=100, validation_data=(X_test, y_test), verbose=0)

    y_probs = model.predict_proba(X_test,verbose=0)
    y_pred = model.predict(X_test)

    y_probs[:,1]
    roc = metrics.roc_auc_score(y_test[:,1], y_probs[:,1])
    auc_scores.append(roc)
    print "Fold Complete", fold

    fpr, tpr, _ = metrics.roc_curve(y_test[:,1], y_probs[:,1])
    annScores.append({'fpr':fpr,'tpr':tpr,'AUC':roc})

plt.plot(annScores[0]['fpr'], annScores[0]['tpr'],c='b')
plt.plot(annScores[1]['fpr'], annScores[1]['tpr'],c='r')
plt.plot(annScores[2]['fpr'], annScores[2]['tpr'],c='g')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ANN ROC curve')
plt.legend(loc="lower right")
plt.show()




print np.mean(auc_scores)

for a in auc_scores:
    print "%.4f" % a

#### Train on all the data


inputD = X.shape[1]
outputD = 2
model = annModel(inputD,outputD)
model.fit(X, yc, nb_epoch=100, batch_size=100, verbose=0)

#### Run it on the test data

test = pd.read_csv('testComb.csv')

# test_weather = test[PCA_cols]
# test2 = test.drop(PCA_cols,axis=1)
#
#
# pca = PCA(n_components=10)
# test_weather_pca = pca.fit_transform(test_weather)
#
#
# test_weather_pca = pd.DataFrame(test_weather_pca)
#
# test = pd.concat([test2,test_weather_pca],axis=1)



sub_probs = model.predict_proba(Xt,verbose=0)

submission = pd.read_csv('../assets/sampleSubmission.csv')

subdat = pd.DataFrame(submission)

subdat.head()

len(submission)
len(sub_probs)

subdat['WnvPresent'] = sub_probs[:,1]

subdat.to_csv('test_sub_ann2.csv', index=False)

subdat.head()



##############
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

bdt = BaggingClassifier(DecisionTreeClassifier(class_weight='balanced'))
rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
et = ExtraTreesClassifier(class_weight='balanced', n_jobs=-1)
abdt = AdaBoostClassifier(DecisionTreeClassifier())
gb = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.0035,loss='deviance',max_features=8,max_depth=7,subsample=1)
ab = AdaBoostClassifier()


kf = KFold(len(y), n_folds=4 ,shuffle=True, random_state=0)

rf_auc_scores = []
# acc_scores = []
rffold = 0

rfScores = []

for training, testing in kf:
    rffold += 1
    print "Fold Start", rffold
    X_train = X[training]
    X_test = X[testing]
    y_train = y[training]
    y_test = y[testing]
    y_true = y[testing]

    clf = gb
    clf.fit(X_train, y_train)
    rf_ypred = clf.predict(X_test)
    rf_yprobs = clf.predict_proba(X_test)

    roc = metrics.roc_auc_score(y_test, rf_yprobs[:,1])
    rf_auc_scores.append(roc)
    print "Fold Complete", rffold

    fpr, tpr, _ = metrics.roc_curve(y_test, rf_yprobs[:,1])
    rfScores.append({'fpr':fpr,'tpr':tpr,'AUC':roc})

rfScores[0][

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])




preds = clf.predict_proba(Xtest)[:,1]
fpr, tpr, _ = metrics.roc_curve(ytest, preds)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')

print np.mean(rf_auc_scores)

for a in rf_auc_scores:
    print "%.4f" % a



clf = gb
clf.fit(X, y)

gb_pred = clf.predict_proba(Xt)


submission = pd.read_csv('../assets/sampleSubmission.csv')
subdat = pd.DataFrame(submission)



len(submission)
len(gb_pred)

subdat['WnvPresent'] = gb_pred[:,1]

subdat

subdat.to_csv('test_sub_gb3.csv', index=False)

subdat.head()

#############

clf = gb
clf.fit(X, y)

gb_pred = clf.predict_proba(Xt)


submission = pd.read_csv('../assets/sampleSubmission.csv')
subdat = pd.DataFrame(submission)



len(submission)
len(gb_pred)

subdat['WnvPresent'] = gb_pred[:,1]

subdat

subdat.to_csv('test_sub_gb3.csv', index=False)
