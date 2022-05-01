import os 
import pandas as pd
import pickle
import 

#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'

os.chdir(path)

# transacciones_previa.to_pickle('Datos/intermedia/transacciones.pkl', compression= 'bz2')

transacciones = pd.read_pickle('Datos/intermedia/transacciones.pkl', compression= 'bz2')


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


#%%

union.to_pickle('Datos/intermedia/union.pkl', compression= 'bz2')

#%%


