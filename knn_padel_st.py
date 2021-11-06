# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:42:44 2021

@author: dguti
"""

import pandas as pd

datos = pd.read_csv("/Users/guill/OneDrive/Escritorio/Master/TFM/base de datos/guardados/Dataset12.csv")

print(datos.shape)
print(datos.info())

#%% eliminamos las columnas que no nos interesan

datos.drop(columns = ["mano", "reves", "altura", "edad", "sexo", "nivel","id", "numero_golpe", "tiempo_golpe"], inplace=True)

#%% nuevos datos que tenemos

print(datos.info())
print(datos.shape)

#%% dividimos los datos

from sklearn.model_selection import train_test_split

X = datos.drop(columns = ["tipo_golpe"])
y = datos["tipo_golpe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


#%% entrenamiento modelo

from sklearn.neighbors import KNeighborsClassifier

K=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30]
#K=[1]
for i in K:
    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(X_train, y_train)

    #%% resultados test
    
    ypred = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    
    print("K = ",i,": ", accuracy_score(y_test, ypred)*100.00) 

#%% matriz de confusión 

#Se muestra la matriz de confusion del ultimo K del bucle. Si se quieren mostrar de todos, meter la llamada a la funcion dentro del bucle for

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, ypred)

golpes = ['D','R','DP','RP','GD','GR','GDP','GRP','VD','VR','B','RM','S']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    plt.xticks(range(13), golpes)
    plt.yticks(range(13), golpes)
    

plt.figure()
plot_confusion_matrix(cm, classes = range(3))  
