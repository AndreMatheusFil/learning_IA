import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


base = pd.read_csv('iris.csv')
previsores= base.iloc[:,0:4].values
classes = base.iloc[:,4].values
labelencoder = LabelEncoder()
classes =labelencoder.fit_transform(classes)
classe_dummy = np_utils.to_categorical(classes)

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units=4,activation="relu",input_dim=4))
    classificador.add(Dense(units=4,activation='relu'))
    classificador.add(Dense(units=3,activation="softmax"))
    classificador.compile(optimizer="adam",loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn= criarRede,
                                epochs = 1000,
                                batch_size = 10)
resultado = cross_val_score(X=previsores, y= classes,estimator=classificador,cv=10,scoring='accuracy')
media = resultado.mean()
desvio = resultado.std()
print(media)
print(desvio)
print(resultado)