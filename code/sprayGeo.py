import pandas as pd

spray = pd.read_csv('../assets/spray.csv')
spray.Date = pd.to_datetime(spray.Date)

spray.Latitude = spray['Latitude'].round(decimals=4)
spray.Longitude = spray['Longitude'].round(decimals=4)

spray.head()

test = pd.read_csv('../assets/test.csv')

test.shape
