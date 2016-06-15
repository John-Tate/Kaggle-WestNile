import pandas as pd
from sklearn.preprocessing import LabelEncoder
from geopy.distance import vincenty

weather = pd.read_csv('../assets/weather.csv')
train = pd.read_csv('../assets/train.csv')
test = pd.read_csv('../assets/test.csv')
spray = pd.read_csv('../assets/spray.csv')
station = pd.DataFrame({'Station':[1,2],'station_geo':[(41.995, -87.933),(41.786, -87.72)]})

test.shape

#weather

# Change Tavg, Depart, WetBulb, Heat, Cool, PrecipTotal, StnPressure, SeaLevel, AvgSpeed to numeric
weather.Date = pd.to_datetime(weather.Date)
weather.Tavg = weather.Tavg.apply(lambda x: int(x) if str(x).isdigit() else None)
weather.Depart = weather.Depart.apply(lambda x: int(x) if str(x).isdigit() else None)
weather.WetBulb = weather.WetBulb.apply(lambda x: int(x) if str(x).isdigit() else None)
weather.Heat = weather.Heat.apply(lambda x: int(x) if str(x).isdigit() else None)
weather.Cool = weather.Cool.apply(lambda x: int(x) if str(x).isdigit() else None)
weather.PrecipTotal = weather.PrecipTotal.apply(lambda x: float(x) if str(x)[0].isdigit() else None)
weather.StnPressure = weather.StnPressure.apply(lambda x: float(x) if str(x)[0].isdigit() else None)
weather.SeaLevel = weather.SeaLevel.apply(lambda x: float(x) if str(x)[0].isdigit() else None)
weather.AvgSpeed = weather.AvgSpeed.apply(lambda x: float(x) if str(x)[0].isdigit() else None)

# Drop Sunrise, Sunset, CodeSum, Depth, Water1
weather = weather.drop(['Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1'], axis=1)

# Change SnowFall to categorical: 0 if no snow/missing, 1 if any snow
def snowcode(x):
    if x == 'M' or x =='0.0':
        return 0
    else: return 1

weather.SnowFall = weather.SnowFall.apply(snowcode)

# train & test

train['Date'] = pd.to_datetime(train['Date'])
train = train.drop(['Address', 'Block', 'Street', 'AddressNumberAndStreet'], axis=1)
train.Species = LabelEncoder().fit_transform(train.Species)
train.Trap = LabelEncoder().fit_transform(train.Trap)
train['Week'] = train.Date.dt.weekofyear


test['Date'] = pd.to_datetime(test['Date'])
test = test.drop(['Address', 'Block', 'Street', 'AddressNumberAndStreet'], axis=1)
test.Species = LabelEncoder().fit_transform(test.Species)
test.Trap = LabelEncoder().fit_transform(test.Trap)
test['Week'] = test.Date.dt.weekofyear

# spray
spray.Date = pd.to_datetime(spray.Date)
spray = spray.drop('Time', axis = 1)

# Identify nearest Weather Station to each point in train & test

# get a geo coordinate pair for each test in train
def getGeos(df):
    test_geo = []
    for index,item in df.iterrows():
        lat = str(item['Latitude'])
        lon = str(item['Longitude'])
        geo = '('+ lat +','+ lon +')'
        test_geo.append({'Latitude':item['Latitude'],'Longitude':item['Longitude'],'test_geo':geo})
    return test_geo

train_geo1 = getGeos(train)
test_geo1 = getGeos(test)
#convert result to dataframe, and drop duplicates
train_geos = pd.DataFrame(train_geo1)
test_geos = pd.DataFrame(test_geo1)

train_geos = train_geos.drop_duplicates()
test_geos = test_geos.drop_duplicates()

train_geos.shape
test_geos.shape


# Define station locations
station1 = station.station_geo[0]
station2 = station.station_geo[1]

# Compute distance to each station, pick smallest distance
def closestStation(df,station1=station1,station2=station2):
    StationDistance = []
    for index,item in df.iterrows():
        test_site = item['test_geo']
        s1Distance = vincenty(station1, test_site).miles
        s2Distance = vincenty(station2, test_site).miles
        if s1Distance > s2Distance:
            StationDistance.append({'test_geo':test_site,'Station':2})
        else:
            StationDistance.append({'test_geo':test_site,'Station':1})
    return StationDistance

def cloestDF(x):
    x2 = closestStation(x)
    x3 = pd.DataFrame(x2)
    return x3

def geoMerge(x,y,z):
    new = pd.merge(x,y,on=['test_geo'])
    final = pd.merge(z,new,on=['Latitude','Longitude'])
    return final

TrainStationDistance = cloestDF(train_geos)
TestStationDistance = cloestDF(test_geos)

train = geoMerge(train_geos,TrainStationDistance,train)
test = geoMerge(test_geos,TestStationDistance,test)

train.shape

# Merge train and weather on date and station.
train = pd.merge(train,weather,on=['Date','Station'])
test = pd.merge(test,weather,on=['Date','Station'])

# train.to_csv('trainComb.csv', ecnoding='utf-8', index=False)
# test.to_csv('testComb.csv', ecnoding='utf-8', index=False)

##### MODELING
from sklearn.cross_validation import train_test_split
from sklearn import metrics

train.columns

feature_cols =  ['Species','NumMosquitos','Week','Station','Tmax','Tmin','Tavg','DewPoint',
                'SeaLevel','ResultSpeed','ResultDir','AvgSpeed']

X = train[feature_cols]
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

rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
rfmod = rf.fit(X_train,y_train)
rf_ypred = rfmod.predict(X_test)
rf_yprobs = rfmod.predict_proba(X_test)


rf_acs = metrics.accuracy_score(y_test,rf_ypred)
rf_cm = metrics.confusion_matrix(y_test,rf_ypred)
rf_cr = metrics.classification_report(y_test,rf_ypred)

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, rf_yprobs[:,1])
rf_roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

print rf_roc_auc

print rf_acs
print rf_cm

print rf_cr
