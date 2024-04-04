import pandas as pd
import numpy as np
import pickle

def encode_categorical(series):
    col = series.name
    series_one_hot = pd.get_dummies(series)
    if -9 in series_one_hot.columns:
        series_one_hot = series_one_hot.drop(labels=-9, axis=1)
    series_one_hot.columns = [col+'_'+str(i) for i in series_one_hot.columns]
    return series_one_hot

def scale_continuous(series):
    series = series.copy()
    mean = series.loc[series!=-9].mean()
    series.loc[series==-9] = mean
    series = (series-mean)/series.std()
    return pd.DataFrame(series)

data = pd.read_csv('Datasets/treatments_2017-2020.csv')
data = data.loc[data.REASON!=7]
data = data.reset_index(drop=True)

categorical_features = ['EMPLOY_D', 'MARSTAT', 'PRIMINC', 'TRNQFLG', 'NOPRIOR', 'PREG', 'METHUSE', 'DETNLF_D',
                        'INHFLG', 'SUB1_D', 'ALCDRUG', 'SERVICES', 'ETHNIC', 'SERVICES_D', 'DETNLF', 'PRIMPAY',
                        'HERFLG', 'IDU', 'REGION', 'SEDHPFLG', 'BENZFLG', 'OPSYNFLG', 'ALCFLG', 'OTCFLG',
                        'HLTHINS', 'DIVISION', 'GENDER', 'AMPHFLG', 'PCPFLG', 'PSOURCE', 'COKEFLG',
                        'PSYPROB', 'RACE', 'ROUTE1','DETCRIM', 'VET','ARRESTS_D', 'MARFLG','STIMFLG','LIVARAG_D', 'SUB1',
                        'BARBFLG', 'EMPLOY', 'LIVARAG', 'ARRESTS', 'EDUC', 'DSMCRIT', 'METHFLG','OTHERFLG', 'FREQ_ATND_SELF_HELP_D', 'FREQ1_D',
                        'FREQ1', 'FREQ_ATND_SELF_HELP',
                        'metropolitanmicropolitanstatis']
continuous_features = ['AGE', 'LOS', 'FRSTUSE1', 'DAYWAIT']


# sampling 100000 rows
# data = data.sample(100000)

df = data[['DISYR', 'STFIPS', 'CBSA', 'REASON']]
for i in range(len(categorical_features)):
    df = pd.concat([df, encode_categorical(data[categorical_features[i]])], axis=1)
for i in range(len(continuous_features)):
    df = pd.concat([df, scale_continuous(data[continuous_features[i]])], axis=1)
del data

state_data = pd.read_csv('acs_5yr_state_normalized.csv')
CBSA_data = pd.read_csv('acs_5yr_CBSA_normalized.csv')
facs_data = pd.read_csv('facilities_by_state_normalized.csv')

train, test = pickle.load(open('split.p', 'rb'))

def load_batch(ind):
    batch = df.loc[ind]
    batch = pd.merge(batch, state_data, how="left", left_on=["STFIPS", "DISYR"], right_on=["fips", "Year"])
    batch = batch.drop(labels=["fips", "Year"], axis=1)
    batch = pd.merge(batch, CBSA_data, how="left", left_on=["CBSA", "DISYR"], right_on=["CBSA", "Year"])
    batch = batch.drop(labels=[ "Year"], axis=1)
    batch = pd.merge(batch, facs_data, how="left", left_on=["STFIPS", "DISYR"], right_on=["state_fips", "year"])
    batch = batch.drop(labels=["state_fips", "year"], axis=1)
    X = batch.values[:, 4:]
    y = batch.REASON.values
    y = np.int16(y==1)
    return X, y