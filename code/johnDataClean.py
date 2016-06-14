import pandas as pd

weather = pd.read_csv('../assets/weather.csv')
train = pd.read_csv('../assets/train.csv')
spray = pd.read_csv('../assets/spray.csv')

train.head()

weather.head()

station = pd.DataFrame({'Station':[1,2],'station_lat':[41.995,41.786],'station_lon':[-87.933,-87.752]})

station_geo = [(41.995, -87.933),(41.786, -87.72)]

station

station['station_geo'] = station_geo
weather2 = pd.merge(weather,station,on=['Station'])

train.head()

train['test_geo'] = train.Latitude.cat(others=[train.Longitude], sep=',')

test_geo = []


for index,item in train.iterrows():
    lat = str(item['Latitude'])
    lon = str(item['Longitude'])
    geo = '('+ lat +','+ lon +')'
    test_geo.append({'Latitude':item['Latitude'],'Longitude':item['Longitude'],'test_geo':geo})


train.drop('test_geo', axis=1, inplace=True)



train_geos = pd.DataFrame(test_geo)


train = pd.merge(train,train_geos,on=['Latitude','Longitude'])

station1 = station.station_geo[0]
station2 = station.station_geo[1]

from geopy.distance import vincenty

StationDistance = []

for index,item in train.iterrows():
    test_site = item['test_geo']
    s1Distance = vincenty(station1, test_site).miles
    s2Distance = vincenty(station2, test_site).miles
    if s1Distance > s2Distance:
        StationDistance.append({'test_geo':test_site,'Station':2})
    else:
        StationDistance.append({'test_geo':test_site,'Station':1})