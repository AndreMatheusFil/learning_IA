import pandas as pd
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras import optimizers

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')


def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units=16, activation="relu", kernel_initializer="random_uniform", input_dim=30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=16, activation="relu", kernel_initializer="random_uniform"))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation='sigmoid'))
    otimizadores = optimizers.Adam(learning_rate=0.01, decay=0.1, clipvalue=0.5)
    classificador.compile(optimizer=otimizadores, loss="binary_crossentropy", metrics=["binary_accuracy"])
    return classificador


classificador = KerasClassifier(build_fn=criarRede,
                                epochs=100,
                                batch_size=10)
resultado = cross_val_score(estimator=classificador,
                            X=previsores, y=classe,
                            cv=10, scoring='accuracy')
print(resultado)