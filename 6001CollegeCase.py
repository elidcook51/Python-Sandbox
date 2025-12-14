import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm

def aic_score(estimator, X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return -model.aic

sfs = SFS(LinearRegression(),
          k_features = 'best',
          forward = True,
          floating = False,
          scoring = aic_score,
          cv = 0)

df = pd.read_excel("C:/Users/ucg8nb/Downloads/college.data.xlsx")
y = df['Reputation']
X = df[['Accept Rate', 'SAT Verb', 'SAT Math', 'Fac/Stud', '% Male']]

sfs.fit(X, y)
print(sfs.k_feature_names_)

