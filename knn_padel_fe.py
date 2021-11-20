# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:42:44 2021

@author: dguti
"""

import pandas as pd

datos = pd.read_csv("/Users/guill/OneDrive/Escritorio/Master/TFM/base de datos/guardados/Dataset12.csv")

#print(datos.shape)
#print(datos.info())

#%% feature engineering, realizamos un nuevo data frame con
# las siguientes características para cada golpe

"""
Ax_mean: valor medio aceleración eje x
Ay_mean: valor medio aceleración eje y
Az_mean: valor medio aceleración eje z

Vx_mean: valor medio velocidad eje x
Vy_mean: valor medio velocidad eje y
Vz_mean: valor medio velocidad eje z

Ax_max: valor máximo aceleración eje x
Ay_max: valor máximo aceleración eje y
Az_max: valor máximo aceleración eje z

Vx_max: valor máximo velocidad eje x
Vy_max: valor máximo velocidad eje y
Vz_max: valor máximo velocidad eje z

Ax_min: valor mínimo aceleración eje x
Ay_min: valor mínimo aceleración eje y
Az_min: valor mínimo aceleración eje z

Vx_min: valor mínimo velocidad eje x
Vy_min: valor mínimo velocidad eje y
Vz_min: valor mínimo velocidad eje z

"""

datos_features = pd.DataFrame()

datos_features["Ax_mean"] = datos.loc[:, "Ax0":"Ax39"].mean(axis=1)
datos_features["Ay_mean"] = datos.loc[:, "Ay0":"Ay39"].mean(axis=1)
datos_features["Az_mean"] = datos.loc[:, "Az0":"Az39"].mean(axis=1)

datos_features["Vx_mean"] = datos.loc[:, "Vx0":"Vx39"].mean(axis=1)
datos_features["Vy_mean"] = datos.loc[:, "Vy0":"Vy39"].mean(axis=1)
datos_features["Vz_mean"] = datos.loc[:, "Vz0":"Vz39"].mean(axis=1)

datos_features["Ax_max"] = datos.loc[:, "Ax0":"Ax39"].max(axis=1)
datos_features["Ay_max"] = datos.loc[:, "Ay0":"Ay39"].max(axis=1)
datos_features["Az_max"] = datos.loc[:, "Az0":"Az39"].max(axis=1)

datos_features["Vx_max"] = datos.loc[:, "Vx0":"Vx39"].max(axis=1)
datos_features["Vy_max"] = datos.loc[:, "Vy0":"Vy39"].max(axis=1)
datos_features["Vz_max"] = datos.loc[:, "Vz0":"Vz39"].max(axis=1)

datos_features["Ax_min"] = datos.loc[:, "Ax0":"Ax39"].min(axis=1)
datos_features["Ay_min"] = datos.loc[:, "Ay0":"Ay39"].min(axis=1)
datos_features["Az_min"] = datos.loc[:, "Az0":"Az39"].min(axis=1)

datos_features["Vx_min"] = datos.loc[:, "Vx0":"Vx39"].min(axis=1)
datos_features["Vy_min"] = datos.loc[:, "Vy0":"Vy39"].min(axis=1)
datos_features["Vz_min"] = datos.loc[:, "Vz0":"Vz39"].min(axis=1)

datos_features["tipo_golpe"] = datos["tipo_golpe"].astype(int)

#%% nuevos datos que tenemos

print(datos_features.info())
print(datos_features.shape)

#%% dividimos los datos

from sklearn.model_selection import train_test_split

X = datos_features.drop(columns = ["tipo_golpe"])
y = datos_features["tipo_golpe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=5)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)


#%% entrenamiento modelo

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

K=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30]

scores = list()

best_accuracy = 0
best_k = 0

for i in K:
    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(X_train, y_train)

    # resultados test
    
    ypred = model.predict(X_test)
    
    score = accuracy_score(y_test, ypred)*100.00
    print("K = %d : %.2f %%"% (i, score)) 
    scores.append(score)
    
    if score>best_accuracy:
        best_accuracy = score
        best_k = i


print('Mejor parámetro = %d, con un accuracy de = %.2f %%' % (best_k, best_accuracy))

from matplotlib import pyplot
pyplot.figure()
pyplot.boxplot(scores)
pyplot.title('Accuracy para diferentes valores de K')
pyplot.ylabel("Accuracy (%)")
pyplot.grid(linestyle='-', linewidth=0.3)

print(scores)

#%% matriz de confusión 

#Se muestra la matriz de confusion para el mejor K

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
    
score = accuracy_score(y_test, ypred)*100.00

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
#plt.savefig('KNN'+str(K)+'k.png')
