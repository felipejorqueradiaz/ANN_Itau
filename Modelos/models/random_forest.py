#%% Importar librerias
import pandas as pd
import numpy as np
import winsound
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import os

os.chdir('C:/Users/Asus/Documents/GitHub/ANN_Itau')
#Crear variable a predecir!

#%% Carga de Datos

datos= pd.read_pickle('Datos/intermedia/union.pkl', compression= 'bz2')
display(datos.sample(15))
#Sacar el train no ms
df=datos[datos.dataset=='train']

#%%
# datitos=datos.head(50)
datitos=datos[datos['id']==2]
print(datitos)

#%% Dividir datos
datitos=datos[datos['id']==50]
#%% Dividir datos


X = datos[~Variable]

y= datos['Variable']

# X_train, X_test, y_train, y_test = train_test_split(
#     features, labels, test_size=0.33, shuffle=True, stratify=labels
# )

#%% Clasificador

rf=RandomForestClassifier()
DecisionTreeClassifier().get_params()


param_grid = [
    {"Selection__percentile": range(5, 101, 5)}
]
param_grid


gs = GridSearchCV(selection_pipeline, param_grid, n_jobs=-1)
gs.fit(X_train, y_train)



minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(df.loc[:, cols_to_scale])
scaled_data = pd.DataFrame(scaled_data, columns=cols_to_scale)
