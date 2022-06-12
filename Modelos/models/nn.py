import os
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler #Undersampling
from sklearn.metrics import classification_report
import ml_metrics as metrics



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
    
    #model = GaussianNB()
    #model.fit(X_train_us, y_train_us)
    
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
# In order to ignore FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from keras.models import Sequential
from keras.layers import Dense



prod='A-A'
rus = RandomUnderSampler(random_state=0)
    
X = train[prod].drop(['id', 'Periodo', 'Target'], axis=1)
y = train[prod]['Target']
    
X_train_us, y_train_us = rus.fit_resample(X, y)


### MODELO Y ENTRENAR 

    #model = GaussianNB()
    #model.fit(X_train_us, y_train_us)
    
X_test = test[prod].drop(['id', 'Periodo', 'Target'], axis=1)
y_test = test[prod]['Target']
id_per = test[prod][['id', 'Periodo']]
 

########################################


#### AQUI CREAMOS LAS VARIABLES IMPORTANTES
##### PARAMETROS PARA EL MODELO
#####
##### CUANTOS MODELOS QUEREMOS CREAR
##### PARA LOS EXPERIMENTOS
##### Crearemos de 1 a n_modelos-1 modelos 
#####(esta var debe ser mayor o igual a 2)
n_modelos = 4

##### cada modelo tendra n_nodos_capa en cada capa oculta
n_nodos_capa = 8
n_features = X.shape[1]
n_classes = y.shape[1]


##########FUNCION PARA CREAR MODELOS
## Una funcion para crear nuesras redes
## que es más general que la vista anteriormente.
def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    def create_model():
        # Create model
        model = Sequential(name=name)
        for i in range(n):
            model.add(Dense(nodes, input_dim=input_dim, activation='relu'))

        model.add(Dense(output_dim, activation='softmax'))
        #model.add(Dense(output_dim))
        # Compile model
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
    
        return model
    return create_model


########################################
print("======GENERALES=======")
print("n_features \t=\t"+str(n_features))
print("n_classes \t=\t"+str(n_classes))
print("n_modelos \t=\t"+str(n_modelos-1))
print("======================\n\n")


########################################
## AQUI GENERO 3 MODELOS ..
models = [create_custom_model(n_features, n_classes, n_nodos_capa, i, 'model_{}'.format(i)) for i in range(1, n_modelos)]


########################################
### Para invocar los modelos de esta lista de modelos
### Se hace de la siguente manera.
for create_model in models:
    ## create_model es solo una instancia de la FUNCION create_custom_model
    ## El objeto instanciado con la red que hemos creado, esta en create_model()
    create_model().summary()
    


#####################################################3
####################################################
#%%
from keras.callbacks import TensorBoard
## Ahora se calcula la matriz de confusion
import sklearn.metrics as metrics


history_dict = {} ##para los valores
matrices = {} ##para las matrices de confusion



# TensorBoard Callback
# Usando TensorBoard con Keras Model.fit ()
# Al entrenar con Model.fit () de Keras, agregar la tf.keras.callbacks.
# TensorBoard llamada tf.keras.callbacks.TensorBoard asegura que los registros
# se creen y almacenen. Además, habilite el cálculo del histograma en cada época
# con histogram_freq=1 (esto está desactivado por defecto)
cb = TensorBoard(log_dir=path, histogram_freq=1)

for create_model in models:    
    
    model = create_model()
    
    print('Model name:', model.name)
    
    history_callback = model.fit(X_train_us, y_train_us,
                                 #batch_size=10,
                                 epochs=400,
                                 verbose=1,
                                 #validation_data=(X_test, Y_test),## esto es analogo a validation_split = 0.1
                                 validation_split = 0.1,
                                 callbacks=[cb])

    
    ##Para cada modelo evaluo con el test set al fin de su entrenamiento
    ##Imprimo los resultados
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(score)
    #var_dump(score)
    
    ### Genero la predicción con el modelo
    y_pred = model.predict(X_test)
    
    #print(y_pred)

    ### La red da números flotantes, no genera enteros!
    ### Para poder interpretar como corresponde los valores, hay dos opciones:
    ### Una que sirve en clases binarias y la siguiente que funciona en cualquier problema. 
    #######  ESTE CODIGO ES PARA CLASES BINARIAS  ########
    ## Este codigo solo funciona bien en clases binarias:
    
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1
            
    # Y una vez binarizadas las columnas, nececito usar la Inversa del OneHotEncoder para volver a una sola columna
    # Con los valores de las clases, eso se hace asi:
    
    #y_pred_1col = pd.DataFrame(enc.inverse_transform(y_pred), columns= ['target'])
    #######  FIN DEL CODIGO PARA CLASES BINARIAS  ########
    
    history_dict[model.name] = [history_callback, model]
    
    
#%%
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

###########
##Se requiere parametro  n_clases!
n_clases = n_classes

for  model_name in matrices:
    print("\n-----")
    print("Metricas modelo: ",model_name)
    print("Precision total:", precision_macro_average(matrices[model_name]))
    print("Recall total:", recall_macro_average(matrices[model_name]))


    print("\nClase precision recall")
    for label in range(n_clases):
        print(f"{label:5d} {precision(label, matrices[model_name]):9.3f} {recall(label, matrices[model_name]):6.3f}")
###########################
######################33
#%%

# y_pred = model.predict(X_test)
# print('-----------\nPRODUCTO {}\n'.format(prod),classification_report(y_test, y_pred),'\n\n')
    



# pred[prod] = model.predict_proba(X_test).T[1]
# real[prod] = y_test
    
    
# #VALIDACION
# X_val = val[prod].drop(['id', 'Periodo', 'Target'], axis=1)
#     id_per_val = val[prod][['id']]
#     valid[prod] = model.predict_proba(X_val).T[1]

# real = pd.concat([real, id_per], axis = 1, ignore_index=True)
# pred = pd.concat([pred, id_per.reset_index(drop = True)], axis = 1, ignore_index=True)
# valid = pd.concat([valid, id_per_val.reset_index(drop = True)], axis = 1, ignore_index=True)

# real.columns = product_list + ['id', 'Periodo']
# pred.columns = product_list + ['id', 'Periodo']
# valid.columns = product_list + ['id']