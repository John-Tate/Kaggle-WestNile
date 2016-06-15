import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

weather = pd.read_csv('./assets/weather.csv')
weather.head()
weather.columns

# Station - Station ID Number
# Date - Date of weather data
# Tmax / Tmin / Tavg - Max, Min, and average temperature that day

import seaborn as sns

weather.Station.unique()


sns.distplot(weather.Tmax)
sns.plt.show()

weather.StnPressure.unique()
weather.Cool = pd.to_numeric(weather.Cool, errors = 'coerce')
weather.Heat = pd.to_numeric(weather.Heat, errors='coerce')
weather.WetBulb = pd.to_numeric(weather.WetBulb, errors='coerce')
weather.Tavg = pd.to_numeric(weather.Tavg, errors='coerce')
weather.corr()


train = pd.read_csv('./assets/train.csv')
train.head()
weather.head()


#Update: this is not a valid merge
twmerge = pd.merge(train,weather, on=['Date'])


#Separate rows where the virus was found, vs negative tests
wnv_true = twmerge[twmerge.WnvPresent == 1]
wnv_false = twmerge[twmerge.WnvPresent == 0]


#find unique values for true and false tests, and common values
wnv_true_unique = {}
wnv_false_unique = {}
wnv_common = {}

for col in wnv_true:
    true = set(wnv_true[col].unique())
    false = set(wnv_false[col].unique())
    false_uni = false - true
    true_uni = true - false
    print true_uni
    common = true & false
    wnv_true_unique[col] = true_uni
    wnv_false_unique[col] = false_uni
    wnv_common[col] = common

wnv_false_unique
wnv_common


#percent for each factor that are common to false and positive virus tests
common_percent = {}
for col in wnv_common:
    falsecount = len(wnv_common[col])
    totalcount = len(twmerge[col].unique())
    common_percent[col] = ((float(falsecount)/float(totalcount))*100)
common_percent

#percent unique to false tests. This will help us find features that DON'T help predict the virus, or
#features that help us predict where the virus isn't. Feature elimination
for col in wnv_false_unique:
    falsecount = len(wnv_false_unique[col])
    totalcount = len(twmerge[col].unique())
    false_percent[col] = ((float(falsecount)/float(totalcount))*100)
twmerge.Cool.unique()
wnv_false_unique['Cool']
