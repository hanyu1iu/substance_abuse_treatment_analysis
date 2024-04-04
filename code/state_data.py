import pandas as pd
import numpy as np

demog = pd.read_csv('Datasets/acs_demographics_5yr.csv', low_memory=False)
econ = pd.read_csv('Datasets/acs_economics_5yr.csv', low_memory=False)
hous = pd.read_csv('Datasets/acs_housing_5yr.csv', low_memory=False)
social = pd.read_csv('Datasets/acs_social_5yr.csv', low_memory=False)

sts = np.array([ 1,  2,  4,  5,  6,  8,  9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
       38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 56, 72])

demog = demog.loc[demog.fips.isin(sts)]
econ = econ.loc[econ.fips.isin(sts)]
hous = hous.loc[hous.fips.isin(sts)]
social = social.loc[social.fips.isin(sts)]

def clean(df):
    df = df[['Year', 'fips'] + list(df.columns[:-4])].copy()
    for i in df.columns[2:]:
        if len(df[i].unique()) == 1:
            df = df.drop(labels=[i], axis=1)
    to_drop = []
    for i in df.columns[2:]:
        series = pd.to_numeric(df[i], errors='coerce')
        series = series - series.mean()
        series = series.fillna(value=0)
        if series.std()>0:
            df[i] = series/series.std()
        else:
            to_drop.append(i)
    df = df.drop(labels=to_drop, axis=1)
    return df

demog = clean(demog)
econ = clean(econ)
hous = clean(hous)
social = clean(social)

df = pd.merge(demog, econ, on=['Year', 'fips'], how='inner')
df = pd.merge(df, hous, on=['Year', 'fips'], how='inner')
df = pd.merge(df, social, on=['Year', 'fips'], how='inner')
df.to_csv('acs_5yr_state_normalized.csv', index=None)