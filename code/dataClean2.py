import pandas as pd
from sklearn.preprocessing import LabelEncoder
from geopy.distance import vincenty

weather = pd.read_csv('../assets/weather.csv')
train = pd.read_csv('../assets/train.csv')
test = pd.read_csv('../assets/test.csv')
# station = pd.DataFrame({'Station':[1,2],'station_geo':[(41.995, -87.933),(41.786, -87.72)]})

test.shape

#weather

#split then merge onto 1 line for each day
weather = weather.drop(['Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1','SnowFall'], axis=1)
weather1 = weather[weather.Station == 1]
weather1 = weather1.drop('Station',axis=1)
weather2 = weather[weather.Station == 2]
weather2 = weather2.drop('Station',axis=1)
weather = pd.merge(weather1,weather2,on=['Date'])

bad_chars = ['M','-','T',' T','  T']
for char in bad_chars:
    weather = weather.replace(char,-99)

weather.head()


train['Date'] = pd.to_datetime(train['Date'])
train['Week'] = train.Date.dt.weekofyear
train['Month'] = train.Date.dt.month
train['Day'] = train.Date.dt.day
train = train.drop(['Address', 'Block', 'Street', 'AddressNumberAndStreet'], axis=1)
train.Species = LabelEncoder().fit_transform(train.Species)
train.Trap = LabelEncoder().fit_transform(train.Trap)

test['Date'] = pd.to_datetime(test['Date'])
test['Week'] = test.Date.dt.weekofyear
test['Month'] = test.Date.dt.month
test['Day'] = test.Date.dt.day
test = test.drop(['Address', 'Block', 'Street', 'AddressNumberAndStreet'], axis=1)
test.Species = LabelEncoder().fit_transform(test.Species)
test.Trap = LabelEncoder().fit_transform(test.Trap)

weather.Date = pd.to_datetime(weather.Date)


train = train.merge(weather, on='Date')

# Merge train and weather on date and station.
train = pd.merge(train,weather,on='Date')
test = pd.merge(test,weather,on=['Date'])

train = train.drop('Date',axis=1)
test = test.drop('Date',axis=1)

train.head()

train = train.ix[:,(train != -99).any(axis=0)]
test = test.ix[:,(test != -99).any(axis=0)]

train.shape
test.shape


train.to_csv('trainComb.csv', ecnoding='utf-8', index=False)
test.to_csv('testComb.csv', ecnoding='utf-8', index=False)

##### MODELING
from sklearn.cross_validation import train_test_split
from sklearn import metrics


X = train.drop(['WnvPresent','NumMosquitos'],axis=1)
y = train['WnvPresent']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)




# LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_classifier = LDA(n_components=4)
ldamod = lda_classifier.fit(X_train, y_train)
lda_ypred = ldamod.predict(X_test)
lda_yprobs =ldamod.predict_proba(X_test)

lda_acs = metrics.accuracy_score(y_test,lda_ypred)
lda_cm = metrics.confusion_matrix(y_test,lda_ypred)
lda_cr = metrics.classification_report(y_test,lda_ypred)

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, lda_yprobs[:,1])
lda_roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

print lda_roc_auc

print lda_acs
print lda_cm
print lda_cr

# Random Forest

from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier
clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=1)
clf.fit(X_train, y_train)
rf_ypred = clf.predict(X_test)
rf_yprobs = clf.predict_proba(X_test)


rf_acs = metrics.accuracy_score(y_test,rf_ypred)
rf_cm = metrics.confusion_matrix(y_test,rf_ypred)
rf_cr = metrics.classification_report(y_test,rf_ypred)

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, rf_yprobs[:,1])
rf_roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

print rf_roc_auc

print rf_acs
print rf_cm

print rf_cr
# create predictions and submission file
predictions = clf.predict_proba(test)[:,1]



rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
rfmod = rf.fit(X_train,y_train)
