import pandas as pd
import numpy as np
import seaborn as sns

#%% Creación de path ..

#### PONER EL PATH A LA CARPETA MADRE ACÁ

#path='C:/Users/Asus/Documents/GitHub/ANN_Itau/'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'
os.chdir(path)


#%%



data = pd.read_csv('Datos/raw/Consumidores.csv',
                       index_col=0)
data = data.drop(['Comuna', 'Ciudad'], axis=1)

data.fillna('No_indica', inplace=True)

#%%

def dum_sign(dummy_col, threshold=0.03):

    # removes the bind
    dummy_col = dummy_col.copy()

    # what is the ratio of a dummy in whole column
    count = pd.value_counts(dummy_col) / len(dummy_col)

    # cond whether the ratios is higher than the threshold
    mask = dummy_col.isin(count[count > threshold].index)

    # replace the ones which ratio is lower than the threshold by a special name
    dummy_col[~mask] = "others"
    return pd.get_dummies(dummy_col, prefix=dummy_col.name)

#%%

#Generamos las dummys en profesion para aquellos con freq. mayor al 3%
data2 = data['Profesion']
prof_dummy = dum_sign(data2).drop('Profesion_others', axis=1)

#%%

#Generamos las dummys para todas las categoricas restantes
cat_cols = ['Edad', 'Renta', 'Segmento_consumidor',
            'Meses_antiguedad', 'Estado_civil', 'Principalidad']
data3 = data.drop(['Profesion'] + cat_cols, axis=1)

dummies = {}
for col in data[cat_cols].columns:
    dummies[col] = pd.get_dummies(data[col], prefix=col, drop_first = True)
    
#%%

#Exportamos
df = pd.concat(dummies.values(), axis=1)
df = pd.concat([df, data3, prof_dummy], axis=1)

df.to_pickle('Datos/intermedia/consumidores.pkl',
                              compression= 'zip')