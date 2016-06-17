import pandas as pd
from sklearn.cross_validation import cross_val_predict,cross_val_score
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../assets/trainComb.csv')

train.head()

blah = train[train.WnvPresent == 1]
blag = train[train.WnvPresent == 0]
len(blah)

len(blag[blag.NumMosquitos > 5])
len(blah[blah.NumMosquitos > 5])

mosqMean = []

for i in train.NumMosquitos:
    if i >= 5:
        mosqMean.append(1)
    else:
        mosqMean.append(0)

feature_cols =  ['Species','Tmax','Trap','Tmin','Tavg','ResultSpeed','AvgSpeed']


X = train[feature_cols]
y = mosqMean

lr = linear_model.LogisticRegression()

scores = cross_val_score(lr,X,y, cv=5)

scores.mean()

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

rfscores = cross_val_score(rf,X,y, cv=5)

rfscores.mean()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_classifier = LDA(n_components=4)
lda_classifier.fit(X_train,y_train)


ldascores = cross_val_score(lda_classifier,X,y, cv=5)

ldascores.mean()



from sklearn.cross_validation import train_test_split
from sklearn import metrics

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)

lda_classifier = linear_model.LinearRegression()
ldamod = lda_classifier.fit(X_train, y_train)
lda_ypred = ldamod.predict(X_test)
# lda_yprobs =ldamod.predict_proba(X_test)

lda_acs = metrics.accuracy_score(y_test,lda_ypred)
lda_cm = metrics.confusion_matrix(y_test,lda_ypred)
lda_cr = metrics.classification_report(y_test,lda_ypred)

# false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, lda_yprobs[:,1])
# lda_roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

print lda_roc_auc

print lda_acs
print lda_cm
print lda_cr
