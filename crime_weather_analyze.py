import pandas as pd
from numpy import log, mean
import datetime as dt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold

df_crime = pd.read_csv("Crimes_2018.csv")
df_crime.head(10)

#pip install holidays

import holidays
us_holidays = holidays.UnitedStates(years=2018)
us_holidays = list(us_holidays.keys())

df_weather = pd.read_csv("climate.csv")
df_weather.head(10)

df_w = df_weather.drop(0)
df_w['DATE'] = df_w['DATE'].astype('str')
df_w['clean_date'] = df_w['DATE'].apply(lambda x: dt.datetime.strptime(x, '%Y%m%d'))
df_w['clean_date'] = df_w['clean_date'].dt.date
df_w = df_w.drop(df_w.columns[[0,1,2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]],1)

df_crime = pd.read_csv("Crimes_2018.csv")
df_crime = df_crime.drop(df_crime.columns[[0,1,3,4,7,14,17,18,22,23,24,25,26,27,28,29]],1)
df_crime.columns = df_crime.columns.str.replace(' ','_',regex=True)
df_crime['intermediary_date'] = pd.to_datetime(df_crime['Date'])
df_crime['clean_date'] = df_crime['intermediary_date'].dt.date
df_crime['clean_time'] = df_crime['intermediary_date'].dt.time
df_crime.sort_values(by=['clean_date'])

df_crime_weather = pd.merge(df_crime,df_w,how='inner',on='clean_date')
df_crime_weather = df_crime_weather.set_index('Date')
df_crime_weather['holiday'] = df_crime_weather['clean_date'].isin(us_holidays) #true indicates holiday

features = ['PRCP', 'SNWD', 'SNOW', 'TAVG', 'AWND']
def normalize_df(df= df_crime_weather,features= features):
    scaler = StandardScaler().fit(df[features])
    df[features] = StandardScaler().fit_transform(df[features])
    return df, scaler
    
normalize_df()

df_crime_weather['season'] = df_crime_weather['clean_date'].apply(lambda dt: (dt.month%12 + 3)//3)

df_crime_weather['shift'] = df_crime_weather['clean_time'].apply(lambda dt: (dt.hour%24 + 8)//8)

df_crime_weather['winter'] = df_crime_weather['season'].apply(lambda x: 1 if x==1 else 0)
df_crime_weather['spring'] = df_crime_weather['season'].apply(lambda x: 1 if x==2 else 0)
df_crime_weather['summer'] = df_crime_weather['season'].apply(lambda x: 1 if x==3 else 0)
df_crime_weather['fall'] = df_crime_weather['season'].apply(lambda x: 1 if x==4 else 0)

df_crime_weather['a_shift'] = df_crime_weather['shift'].apply(lambda x: 1 if x==1 else 0)
df_crime_weather['b_shift'] = df_crime_weather['shift'].apply(lambda x: 1 if x==2 else 0)
df_crime_weather['c_shift'] = df_crime_weather['shift'].apply(lambda x: 1 if x==3 else 0)

df_crime_weather.columns

df_theft_weather = df_crime_weather[df_crime_weather.Primary_Type == 'THEFT']
df_battery_weather = df_crime_weather[df_crime_weather.Primary_Type == 'BATTERY']

df_theft_regression = df_theft_weather.drop(df_theft_weather.columns[[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,23,24,25,26,27,28,29,30,31,32]],1)

df_battery_regression = df_battery_weather.drop(df_battery_weather.columns[[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,23,24,25,26,27,28,29,30,31,32]],1)

df_theft_regression['Arrest'] = df_theft_regression['Arrest'].astype('bool')
df_battery_regression['Arrest'] = df_battery_regression['Arrest'].astype('bool')

def normalize_df(df,features):
    scaler = StandardScaler().fit(df[features])
    df[features] = StandardScaler().fit_transform(df[features])
    return df, scaler

all_features = ['Arrest', 'Domestic', 'Beat','District', 'Ward', 'Community_Area', 'PRCP', 'SNWD','SNOW', 'TAVG', 'AWND', 'holiday', 'season', 'shift', 'winter','spring', 'summer', 'fall', 'a_shift', 'b_shift', 'c_shift']
combo_features = ['PRCP', 'SNWD','SNOW', 'TAVG', 'AWND', 'holiday','winter','spring', 'summer', 'fall', 'a_shift', 'b_shift', 'c_shift']
shift_features = ['a_shift', 'b_shift', 'c_shift']
calendar_features = ['holiday','winter','spring', 'summer', 'fall']
weather_features = ['PRCP', 'SNWD', 'SNOW', 'TAVG', 'AWND']

weather_features = ['PRCP', 'SNWD', 'SNOW', 'TAVG', 'AWND']

def main_fx(features, x, df_crime = df_crime):
    df_weather = pd.read_csv("climate.csv")
    df_weather.head(10)

    df_w = df_weather.drop(0)
    df_w['clean_date'] = df_w['DATE'].apply(lambda x: dt.datetime.strptime(x, '%Y%m%d'))
    df_w['clean_date'] = df_w['clean_date'].dt.date
    df_w = df_w.drop(df_w.columns[[0,1,2,3,4,5,6,7,12,13,15,16,17,18,19,20,21,22,23,24,25,26]],1)

    df_crime = pd.read_csv("Crimes_2018.csv")
    df_crime = df_crime.drop(df_crime.columns[[0,1,3,4,7,14,17,18,22,23,24,25,26,27,28,29]],1)
    df_crime.columns = df_crime.columns.str.replace(' ','_',regex=True)
    df_crime = df_crime[df_crime.Primary_Type == x]
    df_crime['intermediary_date'] = pd.to_datetime(df_crime['Date'])
    df_crime['clean_date'] = df_crime['intermediary_date'].dt.date
    df_crime['clean_time'] = df_crime['intermediary_date'].dt.time
    df_crime.sort_values(by=['clean_date'])

    df_crime_weather = pd.merge(df_crime,df_w,how='inner',on='clean_date')
    df_crime_weather = df_crime_weather.set_index('Date')
    df_crime_weather['holiday'] = df_crime_weather['clean_date'].isin(us_holidays)

    df_crime_weather['season'] = df_crime_weather['clean_date'].apply(lambda dt: (dt.month%12 + 3)//3)
    df_crime_weather['shift'] = df_crime_weather['clean_time'].apply(lambda dt: (dt.hour%24 + 8)//8)

    df_crime_weather['winter'] = df_crime_weather['season'].apply(lambda x: 1 if x==1 else 0)
    df_crime_weather['spring'] = df_crime_weather['season'].apply(lambda x: 1 if x==2 else 0)
    df_crime_weather['summer'] = df_crime_weather['season'].apply(lambda x: 1 if x==3 else 0)
    df_crime_weather['fall'] = df_crime_weather['season'].apply(lambda x: 1 if x==4 else 0)

    df_crime_weather['a_shift'] = df_crime_weather['shift'].apply(lambda x: 1 if x==1 else 0)
    df_crime_weather['b_shift'] = df_crime_weather['shift'].apply(lambda x: 1 if x==2 else 0)
    df_crime_weather['c_shift'] = df_crime_weather['shift'].apply(lambda x: 1 if x==3 else 0)
    def normalize_df(df= df_crime_weather,features= features):
        scaler = StandardScaler().fit(df[features])
        df[features] = StandardScaler().fit_transform(df[features])
        return df, scaler
   
    normalize_df()

    return df_crime_weather

def estimate_probability(the_X, the_model):
    predicted = the_model.predict_proba(the_X)
    return pd.DataFrame(data=predicted, columns=['prob_get_away','prob_arrest'], index=the_X.index)

def calc_log_loss(x):
    if x['Arrest'] == 1:
        return -log(x['prob_arrest'])
    return -log(1-x['prob_arrest'])

def normalize_df(df,features):
    scaler = StandardScaler().fit(df[features])
    df[features] = StandardScaler().fit_transform(df[features])
    return df, scaler

def split_df(df):
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df, df['Arrest'], test_size=0.2, random_state=0)
    return df_X_train, df_X_test, df_y_train, df_y_test

def prediction(df_X_train, df_X_test, df_y_train, df_y_test, features, c=100.0, use_cv=True, coef_df=pd.DataFrame()):
    if use_cv:
        model = LogisticRegressionCV(tol=1.0e-4, penalty='l2', Cs=25, fit_intercept=True, n_jobs=5, cv=StratifiedKFold(n_splits=5),scoring='neg_log_loss', solver='liblinear', refit=True, random_state=0)
    else:
        model = LogisticRegression(tol=1.0e-4, penalty='l2', C=c, fit_intercept=True, warm_start=True, solver='liblinear')
    model.fit(df_X_train[features], df_y_train)

    df_is = estimate_probability(df_X_train[features],model)
    df_X_train = pd.concat([df_X_train,df_is], axis=1, join='outer')
    df_X_train['log_loss'] = df_X_train.apply(calc_log_loss,1)
    log_loss_is = mean(df_X_train.log_loss.values)

    df_oos = estimate_probability(df_X_test[features],model)
    df_X_test = pd.concat([df_X_test,df_oos], axis=1, join='outer')
    df_X_test['log_loss'] = df_X_test.apply(calc_log_loss,1)
    log_loss_oos = mean(df_X_test.log_loss.values)

    if use_cv: c = model.C_[0]
    coef_df = coef_df.append(pd.Series([c,log_loss_is,log_loss_oos,model.intercept_[0]] + model.coef_[0].tolist()),ignore_index=True)
    coef_df.columns = ['c','log_loss_is','log_loss_oos','intercept'] + features

    return df_X_train, df_X_test, coef_df

def run(features, x, use_cv=True):
    df = main_fx(features, x)
    features = features
    df, scaler = normalize_df(df,features)
    df_X_train, df_X_test, df_y_train, df_y_test = split_df(df)
    df_X_train, df_X_test, coef_df = prediction(df_X_train, df_X_test, df_y_train, df_y_test, features, use_cv=use_cv, c=100.0)

    return coef_df

run(combo_features, 'THEFT')
run(combo_features, 'BATTERY')
run(shift_features, 'THEFT')
run(shift_features, 'BATTERY')
run(calendar_features, 'THEFT')
run(calendar_features, 'BATTERY')
run(weather_features, 'THEFT')
run(weather_features, 'BATTERY')