import prepare
import pandas as pd
from sklearn.model_selection import train_test_split
VIRUS_DATA = 'virus_data.csv'
df = pd.read_csv(VIRUS_DATA,header=0)
X_train, X_test, y_train, y_test = train_test_split(df.drop(['spread','risk'],axis=1), df[['spread', 'risk']], test_size=0.2, random_state=12)
train = pd.concat((X_train, y_train), axis=1)
test = pd.concat((X_test, y_test), axis=1)
# Prepare training set according to itself
train_df_prepared = prepare.prepare_data(train, train)
# Prepare test set according to the raw training set
test_df_prepared = prepare.prepare_data(train, test)
VIRUS_DATA_TRAIN = 'virus_data_train.csv'
VIRUS_DATA_TEST= 'virus_data_test.csv'
train_df_prepared.to_csv(VIRUS_DATA_TRAIN)
test_df_prepared.to_csv(VIRUS_DATA_TEST)
