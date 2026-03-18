import pandas as pd
import random
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import pickle

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import math
import joblib
import os
from functools import partial


folderPath = "C:/Users/ucg8nb/Downloads/March Madness Predictor/archive"
teamNameIdCrosswalk = "C:/Users/ucg8nb/Downloads/March Madness Predictor/MTeams.csv"
regularSeasonGames = "C:/Users/ucg8nb/Downloads/March Madness Predictor/MRegularSeasonCompactResults.csv"
tournamentGames = "C:/Users/ucg8nb/Downloads/March Madness Predictor/MNCAATourneyCompactResults.csv"
bigDfPath = "C:/Users/ucg8nb/Downloads/bigDf.csv"
smallTrainingDataPath = "C:/Users/ucg8nb/Downloads/AllGames.csv"
bigTrainingDataPath = "C:/Users/ucg8nb/Downloads/BigGamesTrainingSet.csv"
modelFolderPath = 'C:/Users/ucg8nb/Python Projects/Python-Sandbox/modelStorage'

with open('C:/Users/ucg8nb/Python Projects/Python-Sandbox/modelStorage/tournament_model_features.pkl', 'rb') as f:
    features = pickle.load(f)

trainingData = pd.read_csv(bigTrainingDataPath)
testingData = pd.read_csv(smallTrainingDataPath)

HIDDEN_DIM = (12,4)
# torch.manual_seed(4)

def forward_stepwise_logistic_regression(X, y):
    remaining = list(X.columns)
    selected = []
    current_aic = np.inf
    while len(remaining) > 0:
        aic_with_candidates = []

        for candidate in remaining:
            X_try = sm.add_constant(X[selected + [candidate]])
            model = sm.Logit(y, X_try).fit(disp = False)
            aic_with_candidates.append((model.aic, candidate))

        best_aic, best_candidate = min(aic_with_candidates)
        if best_aic < current_aic:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_aic = best_aic
            print(f"Added {best_candidate}, AIC = {best_aic:.3f}")
        else:
            break
    X_final = sm.add_constant(X[selected])
    final_model = sm.Logit(y, X_final).fit(disp = False)

    return selected, final_model

def testTwoTeams(team0, team1, model):

    currentData = pd.read_csv(bigDfPath)
    currentData = currentData[currentData['Season'] == 2026]

    team0df = currentData[currentData['TeamName'] == team0]
    team1df = currentData[currentData['TeamName'] == team1]
    team0df = team0df.iloc[[0]].reset_index(drop = True)
    team1df = team1df.iloc[[0]].reset_index(drop = True)
    team0df = team0df.add_prefix('Team0')
    team1df = team1df.add_prefix('Team1')

    newDf = pd.concat([team0df, team1df], axis = 1)


    # 2) Create dummies from new data
    Xd = pd.get_dummies(newDf, drop_first=True)

    # 3) Get the exact training column names from the fitted statsmodels object
    train_exog_names = model.model.exog_names  # includes 'const' if you trained with a constant
    feat_no_const = [c for c in train_exog_names if c != 'const']

    # 4) Reindex to the trained features (excluding 'const'), fill any missing with 0
    Xd = Xd.reindex(columns=feat_no_const, fill_value=0)

    # 5) Add constant and reorder columns to match exactly
    X_sm = sm.add_constant(Xd, has_constant='add')
    X_sm = X_sm[train_exog_names]  # ensure exact order and presence

    # 6) Predict
    prob = model.predict(X_sm)

    if prob.empty:
        raise ValueError(f"Didn't work for {team0} and {team1}")

    prob = prob.iloc[0]
    return prob

# ---- HERE IS THE CODE TO ACTUALLY GET THE MODELS -----

def trainLogisticModel(trainingData, outputName):

    columnsToExclude = ['Team1Season', 'Team1TeamName', 'Team0Season', 'Team0TeamName']
    y_col = 'Winner'
    X_cols = [c for c in trainingData.columns if (c != y_col) & (c not in columnsToExclude)]
    X = pd.get_dummies(trainingData[X_cols], drop_first = True)
    y = trainingData[y_col].astype(int)

    # Ensure ints/floats only (optional but safe)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Replace ±Inf with NaN, then drop any column that contains a NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    bad_cols = X.columns[X.isna().any(axis=0)]
    if len(bad_cols) > 0:
        print(f"Dropping columns with NaN/Inf: {list(bad_cols)}")
        X = X.drop(columns=bad_cols)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, stratify=y, random_state=216
    )

    selected_vars, model = forward_stepwise_logistic_regression(X_train, y_train)
    X_test = sm.add_constant(X_test[selected_vars], has_constant = 'add')

    y_proba = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)     # compute ROC
    auc = roc_auc_score(y_test, y_proba)                  # compute AUC

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='navy', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    model.save(f'{outputName}.sm')

    with open(f"{outputName}_features.pkl", 'wb') as f:
        pickle.dump(selected_vars, f)

# --- Class for the MLP (NN on tabular data) ---

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim = (128, 64), dropout = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dim:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        out_dim = 1

        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
# --- Class for the data loader ---
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Helper method which analyzes how good a model is
def evaluate(model, loader, criterion = nn.BCEWithLogitsLoss()):
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    all_preds = []
    all_targets = []
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits.squeeze(1), yb)
        probs = torch.sigmoid(logits.squeeze(1))
        preds = (probs >= 0.5).float()
        correct += (preds == yb).sum().item()
        all_preds.append(probs.cpu())
        all_targets.append(yb.cpu())
    
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    avg_loss = total_loss / n
    accuracy = correct / n
    return {'loss': avg_loss, 'accuracy': accuracy}


def trainNNModel(trainingData, testingData, outputName, cols = None, isPCA = False):
    columnsToExclude = ['Team1Season', 'Team1TeamName', 'Team0Season', 'Team0TeamName']
    y_col = 'Winner'
    X_cols = [c for c in trainingData.columns if (c != y_col) & (c not in columnsToExclude)]

    if cols is not None:
        X_cols = cols

    data_no_nan = trainingData[X_cols + [y_col]].dropna()

    print(f"Removed {len(trainingData) - len(data_no_nan)} rows for having NaN values")

    X_train = data_no_nan[X_cols]
    y_train = data_no_nan[y_col]

    X_test = testingData[X_cols]
    y_test = testingData[y_col]

    if isPCA:
        num_pipeline = Pipeline(
            steps = [('scalar', StandardScaler()), ('pca', PCA(n_components=0.99))]
        )

        preprocessor = ColumnTransformer(
            transformers = [('num', num_pipeline, X_cols)],
            remainder = 'drop'
        )
    else:
        preprocessor = ColumnTransformer(
            transformers = [('num', StandardScaler(), X_cols)],
            remainder= 'drop'
        )
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=216)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.fit_transform(X_test)

    input_dim = X_train_proc.shape[1]

    y_train_t = torch.tensor(y_train.to_numpy(dtype = np.float32), dtype=torch.float32)
    y_test_t = torch.tensor(y_test.to_numpy(dtype = np.float32), dtype=torch.float32)

    X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)

    train_ds = TabularDataset(X_train_t, y_train_t)
    test_ds = TabularDataset(X_test_t, y_test_t)

    batchSize = 128
    train_loader = DataLoader(train_ds, batch_size=batchSize)
    test_loader = DataLoader(test_ds, batch_size = batchSize)

    model = MLP(input_dim = input_dim, hidden_dim = HIDDEN_DIM)

    lr = 1e-3
    weight_decay = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

    epochs = 200
    patience = 25
    best_val = float('-inf')
    best_state = None
    epochs_no_improve = 0

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        running = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.squeeze(1), yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            total += xb.size(0)

        train_loss = running/total
        test_metrics = evaluate(model, test_loader)

        print(f"Epoch{epoch:02d} | train_loss={train_loss:.4f} | test_loss={test_metrics['loss']:.4f} | test_accuracy={test_metrics['accuracy']:.4f}")
        score_to_minimize = test_metrics['accuracy']

        if score_to_minimize > best_val + 1e-6:
            best_val = score_to_minimize
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve > patience:
                print(f"Early Stopping after {epoch} epochs")
                break
        
        
    if best_state is not None:
        model.load_state_dict(best_state)
    
    torch.save(model.state_dict(), f"{outputName}.pt")
    joblib.dump(preprocessor, f"{outputName}_preprocessor.joblib")

# model = sm.load('tournament_model.sm')

def get_model_picks(teamProbPicker):
    startingList = ['Duke', 'Siena', 'Ohio St.', 'TCU', "St. John's", 'Northern Iowa', 'Kansas', 'Cal Baptist', 'Louisville', 'South Florida', 'Michigan St.', 'North Dakota St.', 'UCLA', 'UCF', 'Connecticut', 'Furman', 'Florida', 'Lehigh', 'Clemson', 'Iowa', 'Vanderbilt', 'McNeese', 'Nebraska', 'Troy', 'North Carolina', 'VCU', 'Illinois', 'Penn', "Saint Mary's", 'Texas A&M', 'Houston', 'Idaho', 'Arizona', 'LIU', 'Villanova', 'Utah St.', 'Wisconsin', 'High Point', 'Arkansas', 'Hawaii', 'BYU', 'Texas', 'Gonzaga', 'Kennesaw St.', 'Miami FL', 'Missouri', 'Purdue', 'Queens', 'Michigan', 'UMBC', 'Georgia', 'Saint Louis', 'Texas Tech', 'Akron', 'Alabama', 'Hofstra', 'Tennessee', "SMU", 'Virginia', "Wright St.",'Kentucky', 'Santa Clara', 'Iowa St.', 'Tennessee St.']
    seedDict = {}
    seedOrder = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    for i in range(len(startingList)):
        team = startingList[i]
        seed = seedOrder[i % len(seedOrder)]
        seedDict[team] = seed
    
    outputString = ""
    curList = startingList
    while len(curList) > 1:
        nextList = []
        for i in range(0, len(curList), 2):
            team0 = curList[i]
            team1 = curList[i+ 1]
            seed0 = seedDict[team0]
            seed1 = seedDict[team1]

            print(f"Running Test {team0} vs {team1}")
            
            prob1 = teamProbPicker(team0, team1)
            prob0 = teamProbPicker(team1, team0)
            prob = (prob1 + (1 - prob0)) / 2

            seedDiff = seed0 - seed1

            threshold = 0.5 + 0.032 * seedDiff
            if prob > threshold:
                winner = team1
            else:
                winner = team0
                prob = 1 - prob
            outputString += f"{team0} {seed0} vs. {team1} {seed1}: Pick {winner} with probability {prob * 100:.2f}% \n"
            nextList.append(winner)
        curList = nextList
    return outputString


# ---- HERE IS THE CODE TO CREATE THE TRAINING DATA ---

# bigDf = pd.read_csv(bigDfPath)
# games = pd.read_csv(regularSeasonGames)

# outputDf = pd.DataFrame()
# for index, row in games.iterrows():
#     #Check if the season and team data exists
#     tempDf = bigDf[bigDf['Season'] == row['Season']]
#     if len(tempDf) == 0:
#         continue
#     winningTeam = row['WTeamID']
#     winningDf = tempDf[tempDf['TeamID'] == winningTeam]
#     losingTeam = row['LTeamID']
#     losingDf = tempDf[tempDf['TeamID'] == losingTeam]
#     if len(winningDf) == 0 or len(losingDf) == 0:
#         continue
#     winningDf = winningDf.iloc[[0]].reset_index(drop = True)
#     losingDf = losingDf.iloc[[0]].reset_index(drop = True)
#     if random.random() > 0.5:
#         winningDf = winningDf.add_prefix('Team0')
#         losingDf = losingDf.add_prefix('Team1')
#         winner = 0
#     else:
#         winningDf = winningDf.add_prefix('Team1')
#         losingDf = losingDf.add_prefix('Team0')
#         winner = 1
#     newDf = pd.concat([winningDf, losingDf], axis = 1)
#     if len(newDf) == 0:
#         continue
#     newDf['Winner'] = winner
#     outputDf = pd.concat([outputDf, newDf], axis = 0, ignore_index=True)
# outputDf.to_csv("C:/Users/ucg8nb/Downloads/BigGamesTrainingSet.csv")


# teamCrosswalkDf = pd.read_csv(teamNameIdCrosswalk)
# teamCrosswalkDf = teamCrosswalkDf[['TeamID', 'TeamName']]
# bigDf = pd.read_csv(bigDfPath)
# bigDf = bigDf.merge(teamCrosswalkDf, on = 'TeamName', how = 'left')
# bigDf.to_csv(bigDfPath)

# bigDf = pd.DataFrame()
# for file in os.listdir(folderPath):
#     tempDf = pd.read_csv(os.path.join(folderPath, file))
#     if len(bigDf) == 0:
#         bigDf = tempDf
#     else:
#         bigDf = bigDf.merge(tempDf, how = 'inner', on = ['Season', 'TeamName'])
# bigDf.to_csv(bigDfPath)




# --- All of the NN testing ---
def getNNModelOut(modelPath, preprocessorPath, X_cols):
    preprocessor = joblib.load(preprocessorPath)
    input_dim = len(X_cols)
    model = MLP(input_dim=input_dim, hidden_dim = HIDDEN_DIM)
    state_dict = torch.load(modelPath, map_location = "cpu")
    model.load_state_dict(state_dict)
    return model, preprocessor

def makePredictionWithNNModel(team0, team1, model, preprocessor, X_cols):
    model.eval()

    currentData = pd.read_csv(bigDfPath)
    currentData = currentData[currentData['Season'] == 2026]

    team0df = currentData[currentData['TeamName'] == team0]
    team1df = currentData[currentData['TeamName'] == team1]

    team0df = team0df.iloc[[0]].reset_index(drop = True)
    team1df = team1df.iloc[[0]].reset_index(drop = True)
    team0df = team0df.add_prefix('Team0')
    team1df = team1df.add_prefix('Team1')

    newDf = pd.concat([team0df, team1df], axis = 1)

    X = newDf[X_cols]

    X_proc = preprocessor.transform(X)
    X_t = torch.tensor(X_proc, dtype = torch.float32)

    logits = model(X_t)
    probs = torch.sigmoid(logits.squeeze(1))

    return probs.item()

# trainNNModel(trainingData, testingData, os.path.join(modelFolderPath, 'properTesting'), cols = features)

modelPath = 'C:/Users/ucg8nb/Python Projects/Python-Sandbox/modelStorage/properTesting.pt'
preprocessorPath = 'C:/Users/ucg8nb/Python Projects/Python-Sandbox/modelStorage/properTesting_preprocessor.joblib'

model, preprocessor = getNNModelOut(modelPath, preprocessorPath, X_cols = features)

nnPredFunc = partial(makePredictionWithNNModel, model = model, preprocessor = preprocessor, X_cols = features)

print(get_model_picks(nnPredFunc))

# print(makePredictionWithNNModel('UMBC','Howard', model, preprocessor, features))

