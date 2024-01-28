from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import datetime as dt
import pandas as pd

def transform_data(df):
  # Q12: merge blood types, create one-hot encoding features
  df['blood_type_A'] = df['blood_type'].isin(['A+','A-']).astype(int)
  df['blood_type_AB_B'] = df['blood_type'].isin(['AB+','AB-','B+','B-']).astype(int)
  df['blood_type_O'] = df['blood_type'].isin(['O+','O-']).astype(int)

  # Q13: one-hot encoding for symptoms
  for symptom in ['low_appetite', 'sore_throat', 'cough', 'shortness_of_breath',
    'fever']:
    df[symptom] = df['symptoms'].str.contains(symptom, na=0).astype(int)

  # Sex feature -> numerical
  df['sex'].replace(['M', 'F'], [1, -1], inplace=True)

  # Split current_location given in string format to latitude and longitude
  coordinates = df['current_location'].str.extract("\'(?P<latitude>[-\d\.]+)\',[\s]+\'(?P<longitude>[-\d\.]+)")
  df['latitude'] = coordinates['latitude'].astype('float')
  df['longitude'] = coordinates['longitude'].astype('float')

  # Turn pcr_date into days since the epoch 01/01/0001, this is the epoch matplotlib 3.2.1 works with
  # 01/01/0001 is out of range of pandas datetime type so we have to do a little hack
  epoch_diff = (dt.datetime(1970,1,1) - dt.datetime(1,1,1)).days
  df['pcr_date'] = (
      (pd.to_datetime(df['pcr_date']) - dt.datetime(1970,1,1) ) // pd.Timedelta(days=1)
  ) + epoch_diff

  # Drop the former features and also patient_id since it has no meaning for prediction
  df.drop(['patient_id', 'blood_type', 'current_location', 'symptoms'], axis=1, inplace=True)

def prepare_data(training_data, new_data):
  # Copy the parameter needed to be normalized
  data_copy = new_data.copy(True)
  training_copy = training_data.copy(True)

  # Transform the data
  transform_data(training_copy)
  transform_data(data_copy)

  # Normalization
  transform_min_max = [ "age",
                      "num_of_siblings",
                      "sport_activity",
                      "pcr_date",
                      "happiness_score",
                      "PCR_01",
                      "PCR_02",
                      "PCR_03",
                      "PCR_04",
                      "PCR_05",
                      "PCR_07",
                      "PCR_09",
                      ]

  transform_standartization = ["weight",
                              "household_income",
                              "conversations_per_day",
                              "sugar_levels",
                              "PCR_06",
                              "PCR_08",
                              "PCR_10",
                              "latitude",
                              "longitude",
                              ]


  scaler_normalization = MinMaxScaler(feature_range=(-1,1), copy=False)
  scaler_fitted = scaler_normalization.fit(training_copy[transform_min_max])
  data_copy[transform_min_max] = scaler_fitted.transform(data_copy[transform_min_max])

  standard_normalization = StandardScaler(copy=False)
  standard_fitted = standard_normalization.fit(training_copy[transform_standartization]) 
  data_copy[transform_standartization] = standard_fitted.transform(data_copy[transform_standartization])
  
  return data_copy

