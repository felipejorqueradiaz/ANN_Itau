import os
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler #Undersampling
from sklearn.naive_bayes import GaussianNB #Model
from sklearn.metrics import classification_report
import ml_metrics
#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'


os.chdir(path)

#%%
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



for prod in product_list:
    train[prod] = pd.read_pickle('Datos/final/{}_train.pkl'.format(prod), compression= 'zip')
    test[prod] = pd.read_pickle('Datos/final/{}_test.pkl'.format(prod), compression= 'zip')

#%%

real = pd.DataFrame()
pred = pd.DataFrame()

for prod in product_list:
    rus = RandomUnderSampler(random_state=0)
    
    X = train[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    y = train[prod]['Target']
    
    X_train_us, y_train_us = rus.fit_resample(X, y)
    
    model = GaussianNB()
    model.fit(X_train_us, y_train_us)
    
    X_test = test[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    y_test = test[prod]['Target']
    id_per = test[prod][['id', 'Periodo']]
    
    y_pred = model.predict(X_test)
    print('-----------\nPRODUCTO {}\n'.format(prod),classification_report(y_test, y_pred),'\n\n')
    
    pred[prod] = model.predict_proba(X_test).T[1]
    real[prod] = y_test

real = pd.concat([real, id_per], axis = 1, ignore_index=True)
pred = pd.concat([pred, id_per.reset_index(drop = True)], axis = 1, ignore_index=True)

real.columns = product_list + ['id', 'Periodo']
pred.columns = product_list + ['id', 'Periodo']
#%%


prod_vector = np.array(product_list)
corte = 0.5


for mes in real.Periodo.unique():
    real_temp = target[target['Periodo'] == mes].reset_index(drop=True)
    list_true = real_temp['productos'].to_numpy(copy = True).tolist()
    
 
    pred_temp = pred[pred['Periodo'] == mes]
    d_pred = pred_temp[product_list].to_numpy(copy = True)
    pred_sort_mask = d_pred.argsort()
    d2_pred = np.where(d_pred <= corte, 0, 1)
    v_pred = np.where(d2_pred, prod_vector, 'nulo')
    v_pred = np.take_along_axis(v_pred,pred_sort_mask,axis=1)[:, [4, 3, 2, 1, 0]].tolist()
    v_pred = [[valor for valor in lista if valor!='nulo'] for lista in v_pred]
    print('EL MAP5 para el mes {} es:'.format(mes), metrics.mapk(list_true, v_pred, 5), '\n\n')


list_true = target.reset_index(drop=True)['productos'].to_numpy(copy = True).tolist()

pred_temp = pred.copy()
d_pred = pred_temp[product_list].to_numpy(copy = True)
pred_sort_mask = d_pred.argsort()
d2_pred = np.where(d_pred <= corte, 0, 1)
v_pred = np.where(d2_pred, prod_vector, 'nulo')
v_pred = np.take_along_axis(v_pred,pred_sort_mask,axis=1)[:, [4, 3, 2, 1, 0]].tolist()
v_pred = [[valor for valor in lista if valor!='nulo'] for lista in v_pred]
#np.where(XXXXXX <= corte, 0, 1)
print('EL MAP5 en general es:'.format(mes), metrics.mapk(v_true, v_pred, 5), '\n\n')