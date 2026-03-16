import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import os

folderPath = "C:/Users/ucg8nb/Downloads/March Madness Predictor/archive"
teamNameIdCrosswalk = "C:/Users/ucg8nb/Downloads/March Madness Predictor/MTeams.csv"
regularSeasonGames = "C:/Users/ucg8nb/Downloads/March Madness Predictor/MRegularSeasonCompactResults.csv"
tournamentGames = "C:/Users/ucg8nb/Downloads/March Madness Predictor/MNCAATourneyCompactResults.csv"


teamDf = pd.read_csv(teamNameIdCrosswalk)
for file in os.listdir(folderPath):
    

