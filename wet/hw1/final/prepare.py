import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def count_symptoms(symptoms):
    symptoms_list = symptoms.split(';')
    return len(symptoms_list) if symptoms_list != [''] else 0


def prepare_data(training_data, new_data):
    # Copy the data to avoid modifying the original
    data_preprocessed = new_data.copy()
    training_preprocessed = training_data.copy()
    # Create Special property
    training_preprocessed['SpecialProperty'] = training_preprocessed['blood_type'].isin(['O+', 'B+']).astype(int)
    training_preprocessed.drop('blood_type', axis=1, inplace=True)
    data_preprocessed['SpecialProperty'] = data_preprocessed['blood_type'].isin(['O+', 'B+']).astype(int)
    data_preprocessed.drop('blood_type', axis=1, inplace=True)

    # Create num_symptoms
    # Replace NaN with empty strings
    training_preprocessed['symptoms'] = training_preprocessed['symptoms'].fillna('').astype(str)
    data_preprocessed['symptoms'] = data_preprocessed['symptoms'].fillna('').astype(str)
    # Count the number of symptoms each patient has
    training_preprocessed['num_of_symptoms'] = training_preprocessed['symptoms'].apply(count_symptoms)
    data_preprocessed['num_of_symptoms'] = data_preprocessed['symptoms'].apply(count_symptoms)
    # Drop the 'symptoms' column
    training_preprocessed.drop('symptoms', axis=1, inplace=True)
    data_preprocessed.drop('symptoms', axis=1, inplace=True)

    # Change sex to be 1/0
    training_preprocessed['sex'] = training_preprocessed['sex'].map({'M': 0, 'F': 1})
    data_preprocessed['sex'] = data_preprocessed['sex'].map({'M': 0, 'F': 1})

    # Remove patient id
    training_preprocessed.drop('patient_id', axis=1, inplace=True)
    data_preprocessed.drop('patient_id', axis=1, inplace=True)

    # Handle date - transform it number of days since 0000-01-01
    training_preprocessed['pcr_date'] = pd.to_datetime(training_preprocessed['pcr_date'])
    data_preprocessed['pcr_date'] = pd.to_datetime(data_preprocessed['pcr_date'])
    # Extract year, month, and day
    training_preprocessed['pcr_year'] = training_preprocessed['pcr_date'].dt.year
    training_preprocessed['pcr_month'] = training_preprocessed['pcr_date'].dt.month
    training_preprocessed['pcr_day'] = training_preprocessed['pcr_date'].dt.dayofweek

    data_preprocessed['pcr_year'] = data_preprocessed['pcr_date'].dt.year
    data_preprocessed['pcr_month'] = data_preprocessed['pcr_date'].dt.month
    data_preprocessed['pcr_day'] = data_preprocessed['pcr_date'].dt.dayofweek

    # Remove the original `pcr_date`
    training_preprocessed.drop('pcr_date', axis=1, inplace=True)
    data_preprocessed.drop('pcr_date', axis=1, inplace=True)

    # Split 'current_location' into 'latitude' and 'longitude'
    training_preprocessed[['latitude', 'longitude']] = training_preprocessed['current_location'].str.strip(
        "()").str.replace("'", "").str.split(',', expand=True).astype(float)
    data_preprocessed[['latitude', 'longitude']] = data_preprocessed['current_location'].str.strip("()").str.replace(
        "'", "").str.split(',', expand=True).astype(float)

    # Remove the original 'current_location' column
    training_preprocessed.drop('current_location', axis=1, inplace=True)
    data_preprocessed.drop('current_location', axis=1, inplace=True)

    # normalize chosen columns
    columns_to_standardize = ['longitude', 'latitude', 'weight', 'num_of_siblings', 'happiness_score',
                              'household_income', 'conversations_per_day', 'sugar_levels', 'sport_activity',
                              'num_of_symptoms', 'PCR_07', 'PCR_08', 'PCR_09', 'PCR_10', 'PCR_04']
    columns_to_minmax = ['age', 'pcr_year', 'pcr_month', 'pcr_day', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_05', 'PCR_06']

    # Standardize columns
    scaler_standard = StandardScaler().fit(training_preprocessed[columns_to_standardize])
    data_preprocessed[columns_to_standardize] = scaler_standard.transform(data_preprocessed[columns_to_standardize])
    # MinMax scale columns
    scaler_minmax = MinMaxScaler().fit(training_preprocessed[columns_to_minmax])
    data_preprocessed[columns_to_minmax] = scaler_minmax.transform(data_preprocessed[columns_to_minmax])

    return data_preprocessed
