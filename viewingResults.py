import pandas as pd
import matplotlib.pyplot as plt
import re
import statsmodels.api as sm

# gmmResults = pd.read_csv("C:/Users/ucg8nb/Downloads/Results/GMM_errors_THISONE.csv")

# gmmResults = gmmResults[gmmResults['poison_percent'] != 0]
# gmmResults = gmmResults[gmmResults['poison_percent'] != 100]

# # plt.hist(gmmResults['accuracy'], bins = 20)
# # plt.xlim(90,100)
# # plt.title("Histogram of GMM Accuracy")
# # plt.ylabel("Count of GMM Test Runs")
# # plt.xlabel("Accuracy")
# # plt.show()

# type1 = gmmResults['type_I_percent'].tolist()

# # print(len(type1))
# # print(len([x for x in type1 if x == 0]))

# plt.hist(gmmResults['type_II_percent'], bins = 20)
# plt.xlabel("Type II error")
# plt.ylabel('Count')
# plt.title("Type II error for Gaussian Mixture models")
# plt.show()

# newFilepath = "C:/Users/ucg8nb/Downloads/Results/results_wo_wasserstein_11_07.csv"

# newResults = pd.read_csv(newFilepath, usecols = ['Poison Method', 'Detection Strategy', 'Time (s)'])

# percPoisoned = []
# sampleSize = []
# numTrials = []

# for index, row in newResults.iterrows():
#     poisonName = row['Poison Method']
#     stratName = row['Detection Strategy']
#     match = re.search(r"self\.seed=(\d+),self\.perc=([\d.]+)", poisonName)
#     if match:
#         percPoisoned.append(float(match.group(2)))
#     else:
#         percPoisoned.append(None)
#     match = re.search(r"self\.sample_size=(\d+),self\.num_trials=(\d+)", stratName)
#     if match:
#         sampleSize.append(match.group(1))
#         numTrials.append(match.group(2))
#     elif stratName == 'NonRobustNonBootstrapMDistance':
#         sampleSize.append('NonBootstrap')
#         numTrials.append(None)
#     else:
#         sampleSize.append(None)
#         numTrials.append(None)

# newResults['Percent Poisoned'] = percPoisoned
# newResults['Sample Size'] = sampleSize
# newResults['Num Trials'] = numTrials

# newResults.to_csv("C:/Users/ucg8nb/Downloads/Results/timeModeling.csv")

# Load the data
newResults = pd.read_csv("C:/Users/ucg8nb/Downloads/Results/timeModeling.csv")

# Filter out rows with missing 'Num Trials'
bootstrap = newResults[newResults['Num Trials'].notna()]



# Convert predictors and target to numeric
predCols = ['Percent Poisoned', 'Sample Size', 'Num Trials']
X = bootstrap[predCols].apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(bootstrap['Time (s)'], errors='coerce')

# Drop rows with NaNs
valid_rows = X.notnull().all(axis=1) & y.notnull()
X = sm.add_constant(X[valid_rows])
y = y[valid_rows]

# Fit the model
model = sm.OLS(y, X).fit()
# predicted = model.predict(X)

fitted = model.fittedvalues
residuals = model.resid

# sm.qqplot(y)

# plt.scatter(x = fitted, y = residuals)
# plt.title("Residuals vs fitted")
# plt.show()
print(model.summary())