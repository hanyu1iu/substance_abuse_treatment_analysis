import pandas as pd
import numpy as np

data = pd.read_csv('Datasets/treatments_2017-2020.csv')
data = data.loc[data.REASON!=7]
data = data.reset_index(drop=True)

cbsa = data[['CBSA', 'countycountyequivalent', 'statename']]
cbsa = cbsa.drop_duplicates()

def helper(cts):
    if ',' not in cts:
        return np.array([cts])
    else:
        return np.array([i.strip() for i in cts.split(',')])
    
cbsa.countycountyequivalent = cbsa.countycountyequivalent.apply(helper)
cbsa.statename = cbsa.statename.apply(helper)

cbsa_dict = {}
for code, c, s in cbsa.values:
    combs = []
    for i in c:
        for j in s:
            combs.append(i+', '+j)
    cbsa_dict[code] = combs

demog = pd.read_csv('Datasets/acs_demographics_5yr.csv', low_memory=False)
econ = pd.read_csv('Datasets/acs_economics_5yr.csv', low_memory=False)
hous = pd.read_csv('Datasets/acs_housing_5yr.csv', low_memory=False)
social = pd.read_csv('Datasets/acs_social_5yr.csv', low_memory=False)

def clean(df):
    df = df[['Year', 'geographic_area_name'] + list(df.columns[:-4])].copy()
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

def sum_for_cbsa(acs_df):
    df = pd.DataFrame({'CBSA': np.repeat(cbsa.CBSA, 4), 'Year': np.array([2017, 2018, 2019, 2020] * 599)})
    df = df.reset_index(drop=True)
    _base = pd.DataFrame(columns=[i+'_CBSA' for i in acs_df.columns[2:]])
    for code, year in df.values:
        combs = cbsa_dict[code]
        slice = acs_df.loc[(acs_df.geographic_area_name.isin(combs))&(acs_df.Year==year)]
        _base.loc[len(_base)] = slice.values[:, 2:].sum(axis=0)
    return pd.concat([df, _base], axis=1)

demog = sum_for_cbsa(demog)
econ = sum_for_cbsa(econ)
hous = sum_for_cbsa(hous)
social = sum_for_cbsa(social)

df = pd.merge(demog, econ, on=['Year', 'CBSA'], how='inner')
df = pd.merge(df, hous, on=['Year', 'CBSA'], how='inner')
df = pd.merge(df, social, on=['Year', 'CBSA'], how='inner')
df = df[['Year', 'CBSA']+list(df.columns[2:])]

for i in df.columns[2:]:
    df[i] = (df[i] - df[i].mean())/df[i].std()

df.to_csv('acs_5yr_CBSA_normalized.csv', index=None)