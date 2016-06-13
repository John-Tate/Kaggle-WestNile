import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Kaggle-WestNile/assets/train.csv')
test = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Kaggle-WestNile/assets/test.csv')
submit = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Kaggle-WestNile/assets/sampleSubmission.csv')

len(train)
train.dtypes
'''
Much of this data is geographic:
- Address is the full address - this overlaps with Street and Address, much of the same information is present but has already been cleaned
- Block is also spatial, but allows us to better group similar locations even if we don't know where streets are in relation to each other
- Lat/Long allow map plotting with just the most basic shapefile
- Trap represents which trap the data comes from (if they have a letter after them, they represent a surveillance trap)
- AddressAccuracy: accuracy from the geocoder tool. Only values present are 3 (91 rows), 5 (1807), 8 (4628) and 9 (3980)
- Species represents the type of mosquito found
- Num of mosquitos represents the number of mosquitos found (up to 50)
- WNV present is a boolean value for whether the virus was present in any mosquitos in this trap's collection


train.AddressAccuracy.value_counts()
train.NumMosquitos.value_counts()

train.iloc[0]

len(test)
test.dtypes
#Interesting that test dataset is so much larger than training set
#Added value: ID
#Missing value: WNV presence, num mosquitos
#This is just trap locations. It does not show results, except for species of mosquito.


test.iloc[0]

len(submit)
submit.dtypes
#A series of IDs and Booleans that correspond to the test data
#This is what our deliverable will look like - a list of all the IDs and
#our prediction as to whether or not they will test positive for WNV.
