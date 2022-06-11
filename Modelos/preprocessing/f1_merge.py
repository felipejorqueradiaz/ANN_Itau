import os 
import pandas as pd
import numpy as np
import pickle
 
#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'
os.chdir(path)



#%% Lectura de Transacciones

trans = {}

trans['A-A'] = pd.read_pickle('Datos/intermedia/base_tAA.pkl',
                              compression= 'zip')

'''
trans['B-B'] = pd.read_pickle('Datos/intermedia/base_tBB.pkl',
                              compression= 'zip')
trans['C-D'] = pd.read_pickle('Datos/intermedia/base_tCD.pkl',
                              compression= 'zip')
trans['D-E'] = pd.read_pickle('Datos/intermedia/base_tDE.pkl',
                              compression= 'zip')
trans['E-E'] = pd.read_pickle('Datos/intermedia/base_tEE.pkl',
                              compression= 'zip')

a = trans['A-A']

'''

#%% Lectura de Campañas y Comunicaciones

campanas = pd.read_pickle('Datos/intermedia/campañas.pkl', compression= 'zip')
campanas.drop(['dataset', 'Id_Producto', 'Tipo'], axis=1, inplace=True)

comunicaciones = pd.read_pickle('Datos/intermedia/comunicaciones.pkl', compression= 'zip')
comunicaciones.drop(['Id_Producto', 'Tipo'], axis=1, inplace=True)


#%% Merge



mes_test = 202002

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
    
    base.fillna(0, inplace=True)
    
    ##Construcción del Target

    base['Compra'] = base[['Resultado','Target']].max(axis=1)
    
    base.drop(['Target', 'Resultado'], axis=1, inplace=True)
    
    base['Target'] = base.groupby('id')['Compra'].rolling(3).max().shift(-3).reset_index(0,drop=True)
    
    base.dropna().reset_index(inplace = True)
    '''
    base.to_csv(f'Datos/final/{pt}_base.csv',index=False)
    
    ## Train - Test

    train = base[base['Periodo']<mes_test]
    test = base[(base['Periodo']<202005) & (base['Periodo']>=mes_test)]
    
    train.to_pickle('Datos/final/{}_train.pkl'.format(pt), compression= 'zip')
    test.to_pickle('Datos/final/{}_test.pkl'.format(pt), compression= 'zip')
    '''
#%%

#from imblearn.under_sampling import RandomUnderSampler
#%%
'''
X_train = train.drop('NT', axis=1)
y_train = train['NT']

X_test = test.drop('NT', axis=1)
y_test = test['NT']

#%%

rus = RandomUnderSampler(random_state=0)
X_res, y_res = rus.fit_resample(X_train, y_train)

rus = RandomUnderSampler(random_state=0)
X_tres, y_tres = rus.fit_resample(X_test, y_test)
#%%

#from sklearn.linear_model import LogisticRegression

#%%

clf = LogisticRegression(random_state=0).fit(X_res, y_res)
rf = RandomForestClassifier(max_depth=8, random_state=0).fit(X_res, y_res)
#%%

y_pred_logit = clf.predict(X_tres)
y_pred_rf = rf.predict(X_tres)
#%%

print(confusion_matrix(y_tres, y_pred_logit))
print(confusion_matrix(y_tres, y_pred_rf))

print(clf.score(X_tres, y_tres))
#%%

u2.to_pickle('Datos/intermedia/union.pkl', compression= 'bz2')
'''