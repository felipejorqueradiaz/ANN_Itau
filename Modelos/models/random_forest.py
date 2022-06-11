import os
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler #Undersampling
from sklearn.ensemble import RandomForestClassifier #Model
from sklearn.metrics import classification_report
import ml_metrics as metrics
import seaborn as sns
#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'


os.chdir(path)

target= pd.read_pickle('Datos/final/Target.pkl', compression= 'zip')
#%%

product_list = ['A-A',
                'B-B',
                'C-D',
                'D-E',
                'E-E']


#%% Lectura de Train/Test

train = {}
test = {}
val = {}


for prod in product_list:
    train[prod] = pd.read_pickle('Datos/final/{}_train.pkl'.format(prod), compression= 'zip')
    test[prod] = pd.read_pickle('Datos/final/{}_test.pkl'.format(prod), compression= 'zip')
    val[prod] = pd.read_pickle('Datos/final/{}_val.pkl'.format(prod), compression= 'zip')

#%%

real = pd.DataFrame()
pred = pd.DataFrame()
valid = pd.DataFrame()

for prod in product_list:
    rus = RandomUnderSampler(random_state=0)
    
    X = train[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    y = train[prod]['Target']
    
    X_train_us, y_train_us = rus.fit_resample(X, y)
    
    model = RandomForestClassifier(max_depth=5, random_state=0)
    model.fit(X_train_us, y_train_us)
    
    X_test = test[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    y_test = test[prod]['Target']
    id_per = test[prod][['id', 'Periodo']]
    
    y_pred = model.predict(X_test)
    print('-----------\nPRODUCTO {}\n'.format(prod),classification_report(y_test, y_pred),'\n\n')
    
    pred[prod] = model.predict_proba(X_test).T[1]
    real[prod] = y_test
    
    
    #VALIDACION
    X_val = val[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    id_per_val = val[prod][['id']]
    valid[prod] = model.predict_proba(X_val).T[1]

real = pd.concat([real, id_per], axis = 1, ignore_index=True)
pred = pd.concat([pred, id_per.reset_index(drop = True)], axis = 1, ignore_index=True)
valid = pd.concat([valid, id_per_val.reset_index(drop = True)], axis = 1, ignore_index=True)

real.columns = product_list + ['id', 'Periodo']
pred.columns = product_list + ['id', 'Periodo']
valid.columns = product_list + ['id']

#%%

prod_vector = np.array(product_list)
cortes = np.arange(0,1.01,0.05)

values = []
for corte in cortes:
    for mes in pred.Periodo.unique():
        real_temp = target[target['Periodo'] == mes].reset_index(drop=True)
        list_true = real_temp['productos'].to_numpy(copy = True).tolist()
        
     
        pred_temp = pred[pred['Periodo'] == mes]
        d_pred = pred_temp[product_list].to_numpy(copy = True)
        pred_sort_mask = d_pred.argsort()
        d2_pred = np.where(d_pred <= corte, 0, 1)
        v_pred = np.where(d2_pred, prod_vector, 'nulo')
        v_pred = np.take_along_axis(v_pred,pred_sort_mask,axis=1)[:, [4, 3, 2, 1, 0]].tolist()
        v_pred = [[valor for valor in lista if valor!='nulo'] for lista in v_pred]
        valor = metrics.mapk(list_true, v_pred, 5)
        print('EL MAP5 para el mes {} es:'.format(mes), valor)
        values.append([corte, mes, valor])
    
    
    list_true = target.reset_index(drop=True)['productos'].to_numpy(copy = True).tolist()
    
    pred_temp = pred.copy()
    d_pred = pred_temp[product_list].to_numpy(copy = True)
    pred_sort_mask = d_pred.argsort()
    d2_pred = np.where(d_pred <= corte, 0, 1)
    v_pred = np.where(d2_pred, prod_vector, 'nulo')
    v_pred = np.take_along_axis(v_pred,pred_sort_mask,axis=1)[:, [4, 3, 2, 1, 0]].tolist()
    v_pred = [[valor for valor in lista if valor!='nulo'] for lista in v_pred]
    valor = metrics.mapk(list_true, v_pred, 5)
    values.append([corte, 'general', valor])
    print('EL MAP5 en general es:', valor)

values = pd.DataFrame(values)
values.columns = ['corte', 'periodo', 'map5']
#%%

plot=sns.lineplot(data=values, x="corte", y="map5", hue="periodo")
#%%

corte = 0.5

d_val = valid[product_list].to_numpy(copy = True)
val_sort_mask = d_val.argsort()
d2_val = np.where(d_val <= corte, 0, 1)

v_val = np.where(d2_val, prod_vector, 'nulo')
v_val = np.take_along_axis(v_val,val_sort_mask,axis=1)[:, [4, 3, 2, 1, 0]].tolist()
v_val = [[valor for valor in lista if valor!='nulo'] for lista in v_val]
final = [' '.join(row).strip() for row in v_val]

#%%

pred_final = pd.DataFrame()
pred_final['id']=valid['id'].copy()
pred_final['productos'] = final
pred_final['id']=pred_final['id'].astype(np.int64)
pred_final=pred_final.fillna(" ")
pred_final.to_csv('Datos/output/NaibeBayes.csv',index=False)
