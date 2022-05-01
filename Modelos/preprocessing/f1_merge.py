import os 
import pandas as pd

#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'

os.chdir(path)

transacciones=pd.read_csv('Datos/intermedia/transacciones_v1.csv')
campanas=pd.read_csv('Datos/intermedia/campañas.csv')
# Suceptiblilidad 
# Delay a 3 meses
# Correr código en comunicaciones 



#%%

display(transacciones.columns)
display(campanas.head(3))

#%%

union=pd.merge(transacciones,campanas, on=['id', 'Id_Producto', 'Tipo', 'Producto-Tipo','dataset', 'Periodo'],how='left')
display(union)
display(union.columns)


# 'id', 'Id_Producto', 'Tipo','Producto-Tipo', 'Canal', 'Periodo','dataset'

# 'id', 'Id_Producto', 'Tipo', 'Producto-Tipo','dataset', 'Periodo'
#%%

datasample=union.iloc[:800,]

#%%


