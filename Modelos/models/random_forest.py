import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df= pd.read_pickle('Datos/intermedia/union.pkl', compression= 'bz2')

rf=RandomForestClassifier()

#%%
os.getcwd()
