import os 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
 
#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'
os.chdir(path)

#%% Lectura de Transacciones

trans = {}

trans['A-A'] = pd.read_pickle('Datos/intermedia/base_tAA.pkl',
                              compression= 'zip')
trans['B-B'] = pd.read_pickle('Datos/intermedia/base_tBB.pkl',
                              compression= 'zip')
trans['C-D'] = pd.read_pickle('Datos/intermedia/base_tCD.pkl',
                              compression= 'zip')
trans['D-E'] = pd.read_pickle('Datos/intermedia/base_tDE.pkl',
                              compression= 'zip')
trans['E-E'] = pd.read_pickle('Datos/intermedia/base_tEE.pkl',
                              compression= 'zip')

a = trans['A-A']
#%% Lectura de Campañas y Comunicaciones

campanas = pd.read_pickle('Datos/intermedia/campañas.pkl', compression= 'zip')
campanas.drop(['dataset', 'Id_Producto', 'Tipo'], axis=1, inplace=True)

comunicaciones = pd.read_pickle('Datos/intermedia/comunicaciones.pkl', compression= 'zip')
comunicaciones.drop(['Id_Producto', 'Tipo'], axis=1, inplace=True)

#consumidores = pd.read_pickle('Datos/intermedia/consumidores.pkl', compression= 'zip')
#%% Merge

mes_test = 202002
target2 = {}

for pt, data in trans.items():
    
    ## MERGE
    
    data = data.drop_duplicates(subset = ['id', 'Periodo'] ,keep='first')
    base = data.merge(
        campanas[campanas['Producto-Tipo'] == pt]
            .drop('Producto-Tipo', axis=1),
        how='left'
        )
    
    base = base.merge(
        comunicaciones[comunicaciones['Producto-Tipo'] == pt]
            .drop('Producto-Tipo', axis=1),
        how='left'
        )
    
    #base = base.merge(consumidores, how='left', on='id')
    
    base.fillna(0, inplace=True)
    
    ##Construcción del Target

    base['Compra'] = base[['Resultado','Target']].max(axis=1)
    
    base.drop(['Target', 'Resultado'], axis=1, inplace=True)
    
    base['Target'] = base.groupby('id')['Compra'].rolling(3).max().shift(-3).reset_index(0,drop=True)
    
    base.dropna().reset_index(inplace = True)
    
    
    # ACÁ SE GENERAN LOS DATOS PARA EL PANEL DE POWER BI
    #base.to_csv(f'Datos/final/{pt}_base.csv',index=False)
    
    scalador=StandardScaler()
    base['Monto']=scalador.fit_transform(base[['Monto']])
    
    n_target = base[['id', 'Periodo']]
    n_target[pt] = base.groupby('id')['P PT'].rolling(3).max().shift(-3).reset_index(0,drop=True)
    target2[pt] = n_target
    ## Train - Test

    train = base[base['Periodo']<mes_test]
    test = base[(base['Periodo']<202005) & (base['Periodo']>=mes_test)]
    val = base[base['Periodo']==202007]
    
    train.to_pickle('Datos/final/{}_train.pkl'.format(pt), compression= 'zip')
    test.to_pickle('Datos/final/{}_test.pkl'.format(pt), compression= 'zip')
    val.to_pickle('Datos/final/{}_val.pkl'.format(pt), compression= 'zip')
    
#%%

# ACÁ GENERAMOS UN ARCHIVO FINAL PILOTO DE TESTEO
# SINCERAMENTE, NO SIRVE DE NADA, PERO QUERIAMOS VER SI FUNCIONABA EL MAPE
# RECOMENDAMOS SEGUIR CORRIENDO ESTO :D PARA QUE NO SE MUERAN LOS OTROS CODIGOS

ndata = target2['A-A'].merge(target2['B-B'],
                             on = ['id', 'Periodo'],
                             how = 'outer')
ndata = ndata.merge(target2['C-D'],
                             on = ['id', 'Periodo'],
                             how = 'outer')
ndata = ndata.merge(target2['D-E'],
                             on = ['id', 'Periodo'],
                             how = 'outer')
ndata = ndata.merge(target2['E-E'],
                             on = ['id', 'Periodo'],
                             how = 'outer')

ndata.sort_values(['id', 'Periodo'], inplace=True)
#%%
product_list = ['A-A',
                'B-B',
                'C-D',
                'D-E',
                'E-E']

probs = ndata[product_list].reset_index(drop=True)
sort_mask = probs.to_numpy(copy=True).argsort()
compras = np.where(probs >0, 1, 0)
compras = np.where(compras, np.array(product_list), 'nulo')
compras_sorted = np.take_along_axis(compras,sort_mask,axis=1)[:, [4, 3, 2, 1, 0]]

#%%

compras2 = compras_sorted.tolist()
compras2 = [[valor for valor in lista if valor!='nulo'] for lista in compras2]

target_final = ndata[['id', 'Periodo']].copy()
target_final['productos'] = compras2

target_final.to_pickle('Datos/final/Target.pkl', compression='zip')