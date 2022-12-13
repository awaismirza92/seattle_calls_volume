import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import os


# Data address 
address = '../'


# Downloading calls data
if not os.path.exists(address + 'Seattle_Real_Time_Fire_911_Calls.csv'):
    os.system(f"wget -O {address}Seattle_Real_Time_Fire_911_Calls.csv https://data.seattle.gov/api/views/kzjm-xkqj/rows.csv?accessType=DOWNLOAD")


# Reading
print('Reading calls data...')
df = pd.read_csv(address + 'Seattle_Real_Time_Fire_911_Calls.csv',
                 parse_dates=[2], 
                 infer_datetime_format=True)

# Filtering
df =  df[(df.Datetime >= dt.datetime(2004, 1, 1)) & 
         (df.Datetime <= dt.datetime(2021, 12, 31))]


# Feature extraction
print('Extracting features...')
df_interest = df[['Datetime']][df.Datetime >= dt.datetime(2016, 1, 1)]

df_interest = df_interest.groupby([df_interest.Datetime.dt.date, df_interest.Datetime.dt.hour]).count()
df_interest = df_interest.rename(columns={'Datetime':'Volume'})

df_interest.index = df_interest.index.rename(['Date', 'Hour'])
df_interest = df_interest.reset_index()

df_interest['Date'] = pd.to_datetime(df_interest.Date)
df_interest['Month'] = df_interest.Date.dt.month
df_interest['Weekday'] = df_interest.Date.dt.dayofweek

us_cal = USFederalHolidayCalendar()
us_cal_freq = pd.tseries.offsets.CustomBusinessDay(calendar=us_cal)
work_days = pd.date_range(start="1/1/2016",end="12/09/2022", freq=us_cal_freq)
df_interest['Holiday'] = df_interest['Date'].apply(lambda x: 0 if (x in work_days) else 1)

df_interest = df_interest[['Date', 'Month', 'Weekday', 'Hour', 'Holiday', 'Volume']]



## Downloading weather data

if not os.path.exists(address + 'weather_2016_2018.csv'):
    os.system(f"wget -O {address}weather_2016_2018.csv https://www.ncei.noaa.gov/orders/cdo/3169935.csv")

if not os.path.exists(address + 'weather_2019_2021.csv'):
    os.system(f"wget -O {address}weather_2019_2021.csv https://www.ncei.noaa.gov/orders/cdo/3169979.csv")

if not os.path.exists(address + 'ghcnd-stations.txt'):
    os.system(f"wget -O {address}ghcnd-stations.txt http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt")


# Reading weather data
print('Reading weather data...')
df_weather = pd.read_csv(address + 'weather_2016_2018.csv')
df_weather = df_weather.append(pd.read_csv(address + 'weather_2019_2021.csv'))
df_weather['DATE'] = pd.to_datetime(df_weather['DATE'])


# Reading weather stations data
df_stations = pd.read_csv(address + 'ghcnd-stations.txt', 
                          on_bad_lines='skip',
                          sep='\s+',
                          names=['Id', 'Latitude', 'Longitude', 'Elevation', 'Address1', 'Address2', 'Address3', 'Address4'])

# Obtaining station with most weather observations
df_station_tacoma = df_stations[df_stations.Id == df_weather.STATION.value_counts().index[0]]


# Filtering & cleaning data of Tacoma weather station
df_weather_seattle = df_weather[df_weather.STATION.isin(df_station_tacoma.Id)].copy()
df_weather_seattle['DATE'] = pd.to_datetime(df_weather_seattle['DATE'])
df_weather_seattle = df_weather_seattle.rename(columns={'DATE':'Date'})
df_weather_seattle = df_weather_seattle.drop(columns=['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT11'])
df_weather_seattle = df_weather_seattle.dropna()


# Merging calls data with weather data
df_interest_weather = pd.merge(df_interest, df_weather_seattle[['Date', 'AWND', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']], 
                                       on=['Date'])

# Removing precipitation feature because of its low correlation with calls volume
df_interest_weather = df_interest_weather.drop(columns=['PRCP'])



# Test & train split
print('Splitting data into test & train datasets...')
X = df_interest_weather.drop(columns=['Volume'])

X_train = X[(X.Date >= dt.datetime(2016, 1, 1)) &
            (X.Date <= dt.datetime(2020, 12, 31))]
X_train = X_train.drop(columns=['Date'])

X_test =  X[(X.Date >= dt.datetime(2021, 1, 1)) & 
            (X.Date <= dt.datetime(2021, 12, 31))]
X_test = X_test.drop(columns=['Date'])



y = df_interest_weather[['Date', 'Volume']]

y_train = y[(X.Date >= dt.datetime(2016, 1, 1)) &
            (X.Date <= dt.datetime(2020, 12, 31))]
y_train = y_train['Volume']

y_test =  y[(y.Date >= dt.datetime(2021, 1, 1)) & 
            (y.Date <= dt.datetime(2021, 12, 31))]
y_test = y_test['Volume']



# Gradient boost regressor
print('Fitting gradient boost regressor...')
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
y_hat = reg.predict(X_test)
print('Gradient boost regressor results: ')
print(f"R\N{SUPERSCRIPT TWO}: %.4f" % reg.score(X_test, y_test))
print("RMSE: %.4f" % np.sqrt(np.mean((y_hat - y_test) ** 2)))
print("MAE: %.4f" % np.mean(np.absolute((y_hat - y_test))))

print('\n')


# XGBoost regressor
print('Fitting XGBoost regressor...')
xgbr = xgb.XGBRegressor(objective='reg:squarederror') 
xgbr.fit(X_train, y_train)
y_hat = xgbr.predict(X_test)
print('XGBoost regressor results: ')
print(f"R\N{SUPERSCRIPT TWO}: %.4f" % xgbr.score(X_test, y_test)  )
print("RMSE: %.4f" % np.sqrt(np.mean((y_hat - y_test) ** 2)))
print("MAE: %.4f" % np.mean(np.absolute((y_hat - y_test))))



# Actual vs predicted calls plot (over a week)


days = 7
hours = days * 24

df_interest_weather['Datetime'] = pd.to_datetime(df_interest_weather['Date'].astype(str) + 
                                         df_interest_weather['Hour'].astype(str), 
                                         format='%Y-%m-%d%H')

y_hat = xgbr.predict(X_test)

plt.figure(dpi = 150)
plt.plot(df_interest_weather.Datetime[df_interest_weather.Datetime >= dt.datetime(2021, 1, 1)][:hours], 
         y_hat[:hours], label='Actual')
plt.plot(df_interest_weather.Datetime[df_interest_weather.Datetime >= dt.datetime(2021, 1, 1)][:hours], 
         y_test[:hours].values, label='Predicted')

plt.xlabel('Datetime')
plt.ylabel('Calls')

plt.xticks(rotation = 45)
plt.legend()
plt.tight_layout()

print('Plotting actual vs predicted calls over a week...')
plt.savefig('actual_vs_predicted_week.png');



# Actual vs predicted calls plot (over a month)

days = 30
hours = days * 24

df_interest_weather['Datetime'] = pd.to_datetime(df_interest_weather['Date'].astype(str) + 
                                         df_interest_weather['Hour'].astype(str), 
                                         format='%Y-%m-%d%H')
y_hat = xgbr.predict(X_test)

plt.figure(dpi = 150)
plt.plot(df_interest_weather.Datetime[df_interest_weather.Datetime >= dt.datetime(2021, 1, 1)][:hours], 
         y_hat[:hours], label='Actual')
plt.plot(df_interest_weather.Datetime[df_interest_weather.Datetime >= dt.datetime(2021, 1, 1)][:hours], 
         y_test[:hours].values, label='Predicted')

plt.xlabel('Datetime')
plt.ylabel('Calls')

plt.xticks(rotation = 45)
plt.legend()
plt.tight_layout()

print('Plotting actual vs predicted calls over a month...')
plt.savefig('actual_vs_predicted_month.png');

print('Done.')
