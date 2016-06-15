import pandas as pd

weather = pd.read_csv('../assets/weather.csv')
train = pd.read_csv('../assets/train.csv')
spray = pd.read_csv('../assets/spray.csv')


station = pd.DataFrame({'Station':[1,2],'station_geo':[(41.995, -87.933),(41.786, -87.72)]})

test_geo = []

for index,item in train.iterrows():
    lat = str(item['Latitude'])
    lon = str(item['Longitude'])
    geo = '('+ lat +','+ lon +')'
    test_geo.append({'Latitude':item['Latitude'],'Longitude':item['Longitude'],'test_geo':geo})

train_geos = pd.DataFrame(test_geo)
train_geos = train_geos.drop_duplicates()


station1 = station.station_geo[0]
station2 = station.station_geo[1]

from geopy.distance import vincenty

StationDistance = []

for index,item in train_geos.iterrows():
    test_site = item['test_geo']
    s1Distance = vincenty(station1, test_site).miles
    s2Distance = vincenty(station2, test_site).miles
    if s1Distance > s2Distance:
        StationDistance.append({'test_geo':test_site,'Station':2})
    else:
        StationDistance.append({'test_geo':test_site,'Station':1})

StationDistance = pd.DataFrame(StationDistance)

train_geos = pd.merge(train_geos,StationDistance,on=['test_geo'])

train = pd.merge(train,train_geos,on=['Latitude','Longitude'])

train.head()

train.Date = pd.to_datetime(train.Date)
spray.Date = pd.to_datetime(spray.Date)


#Does is spray messing with out data?


#add week and year columns to train and spray
train['Week'] = train.Date.dt.weekofyear
train['Year'] = train.Date.dt.year
train.head()

spray['Week'] = spray.Date.dt.weekofyear
spray['Year'] = spray.Date.dt.year

#create a binary for a sprayed location
spray['Spray_True'] = 1



#get rid of time
spray = spray.drop('Time',axis=1)
train2 = train.copy()

train2.head(1)
spray.head(1)

#round our geo coordinates. At this Latitude, should give us ~80m of accuracy
train2.Latitude = train2['Latitude'].round(decimals=3)
train2.Longitude = train2['Longitude'].round(decimals=3)

spray.Latitude = spray['Latitude'].round(decimals=3)
spray.Longitude = spray['Longitude'].round(decimals=3)


#check 0, 1 and 2 week previous
train3 = train2.copy()
train3['Week'] = train3['Week']-1
train4 = train2.copy()
train4['Week'] = train4['Week']-2

trainspray = pd.merge(train2,spray, how='left', on=['Week','Year','Latitude','Longitude'])
trainspray1 = pd.merge(train3,spray, how='left', on=['Week','Year','Latitude','Longitude'])
trainspray2 = pd.merge(train4,spray, how='left', on=['Week','Year','Latitude','Longitude'])



trainspray

spray2Week = len(trainspray2[trainspray2.Spray_True == 1])
spray1Week = len(trainspray1[trainspray1.Spray_True == 1])
spray0Week = len(trainspray[trainspray.Spray_True == 1])

totalsprayconfound = spray2Week + spray1Week + spray0Week
totalsprayconfound
