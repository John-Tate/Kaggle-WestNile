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