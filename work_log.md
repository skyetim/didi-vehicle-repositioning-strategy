# 2020.10.16
The file **EDA_activetime_dist.ipynb** contains 3 part:
- load data from sql
- EDA (trip time, trip distance, trip fare, pick-up and drop-off locations, first pick-up and last drop-off plot time, active time)
- predict active time distribution

- Change first pick-up and last drop-off plot interval from 1 hour to 15 mins
- In the pick-drop distribution plot
	- change title: add "first"
	- change y-axis to percentage
- Put all data cleaning in one chunk
- Replot all EDA plot using cleaned data, remove boxplots
- Add analysis and detail about data cleaning for trip time, trip distance, trip 

## Todo
May increase the sample size of driver. Currently is 15% of drivers in June 2013


# 2020.10.14
- EDA
- Find the shift of driver, the start and end time of their work

- Sampled 5000 drivers in June 2013. (15% of drivers)
## data cleaning
	- Trip time (Most within 20 mins, only 2 > 5h, based on the time and distance, system mistake)
		- Drop - 8033 (0.38%) records: drop-off time before pick-up time
		- Recalculate trip time  - 929470 (43.46%) records: drop-off time - pick-up time != trip_time_in_secs,  21724 (1.02%) records: the difference > 30s
		- ?Drop - 15381 (0.72%) records: trip time less than 1 minutes
		- ?Drop - 132 (0.01%) records: trip time over 120 minutes
	
	- Trip distance (Most within 20 miles, only 2 > 100mile, travel 100 mile in less than half an hour, system mistake)
		- 447701 (21.01%) records: trip distance within 1 mile
		- 2121917 (99.59%) records: trip distance within 20 mile
	- Trip fare (Most within 20 dollars)
		- Highest is 500, but the travel time is impossible, just a few seconds

	- Try different K, k is the interval between two trips: next pickup - previous drop-off
		- k=3, 63 (0.003%) active time > 24h
		- k=4, 116 (0.005%) active time > 24h
		- k=5, 186 (0.009%) active time > 24h
	- Active time
		- Because of the cutoff, and system mistake: Start before the end of the previous trip



# 2020.10.09
CAPSTONE Meeting with Bo, Tian 2020.10.09
task: weekend moment utilization estimation

- Download data: https://databank.illinois.edu/datasets/IDB-9610843
	- NYC yellow taxi fare and trip dataset from 2010-2013. Location info is stored as latitude and longitude.
	- The provider cleaned the dataset and drop 7.5%
			- distance check
			- time uniformity
			- missing GPS
- Plot distribution of the starting time throughout the day
- Plot distribution of the ending time throughout the day
- Plot distribution of the ending - starting time throughout the day
- Use 2013.01.04-2013.01.08 data, because cannot fit in the memory.
- Calculate the work hour of one shift. work hour = end time - start time