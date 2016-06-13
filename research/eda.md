Describe the data.
What does it represent?
What types are present?
What does each data points' distribution look like?
What kind of cleaning is needed?
Document any potential issues that will need to be resolved.

Spray:

The spray table contains the dates and locations at which the city of Chicago did spraying to kill mosquitos.  For each spraying event, we have the date, time, latitude and longitude of the spray.

Latitude and Longitude are saved as floats, date and times are objects.

Latitude is not normally distributed at all.  Longitude may be a very noisy normal distribution.  However, the distributions of these features is probably not going to be a major concern for our analysis.  We will mostly be using the latitude and longitude to join with the test set, rather than treating them as a predictive feature.

All sprays occur during July, August, and September, with the bulk in August, and all sprays occur during the evening.  These distributions seem close to normal, but once again, this is unlikely to be critical to our analysis.  I doubt we will use the time at all, but even if we do, we will just be using it to join with our main dataframe.  The only feature we will likely see from this dataframe is a binary categorical for whether or not spraying occurred at a given date and time.

The only cleaning required is to convert date and time to datetimes.

The only potential issue is that we have 584 missing values for time.  However, since we may not use time, this is not too severe of a concern.

Train:

There are about 10,000 data points in this set, representing collection results from various mosquito traps.

Much of this data is geographic:
- Address is the full address - this overlaps with Street and Address, much of the same information is present but has already been cleaned (all objects)
- Block is also spatial, but allows us to better group similar locations even if we don't know where streets are in relation to each other (integer)
- Lat/Long allow map plotting with just the most basic shapefile (floats)

Non-spatial data includes:
- Trap represents which trap the data comes from (if they have a letter after them, they represent a surveillance trap)
- AddressAccuracy: accuracy from the geocoder tool. Only values present are 3 (91 rows), 5 (1807), 8 (4628) and 9 (3980) (Integer)
- Species represents the type of mosquito found (Object)
- Num of mosquitos represents the number of mosquitos found (up to 50) (integer)
- WNV present is a dummy value for whether the virus was present in any mosquitos in this trap's collection (integer)

- Longitude is the only data with even a semi normal distribution
- WNV is only present in about 5 percent of cases
- only potential correlation jumping out visibly is latitude/longitude

Test:

- Interesting that test dataset is so much larger than training set (over 116,000)
- Added value: ID
- Missing values: WNV presence, num mosquitos
- This is just data for trap locations. It does not show results, except for species of mosquito.

Sample Submission:

- A series of IDs and Booleans that correspond to the test data
- This is what our deliverable will look like - a list of all the IDs and our prediction as to whether or not they will test positive for WNV.

Map Data:

- From Open Streetmap. Primarily for use in visualizations. 
