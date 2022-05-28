import os 
import pandas as pd
import pickle

#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'

os.chdir(path)

# transacciones_previa.to_pickle('Datos/intermedia/transacciones.pkl', compression= 'bz2')

transacciones= pd.read_pickle('Datos/intermedia/union.pkl', compression= 'bz2')


campanas=pd.read_csv('Datos/intermedia/campañas.csv')
# 

comunicaciones=pd.read_csv('Datos/intermedia/comunicaciones.plk', compression= 'bz2')



#%%

output <- summary(transacciones)
print(output)

#%% Merge

u1=pd.merge(transacciones,campanas, on=['id', 'Id_Producto', 'Tipo', 'Producto-Tipo','dataset', 'Periodo'],how='left')
u2=pd.merge(u1,comunicaciones, on=['id', 'Id_Producto', 'Tipo', 'Producto-Tipo','dataset', 'Periodo'],how='left')


#%%

u2.to_pickle('Datos/intermedia/union.pkl', compression= 'bz2')

#%%

a=transacciones.sample(100)


