import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
estrutura_rede= arquivo.read()
arquivo.close()

classificador= model_from_json(estrutura_rede)
classificador.load_weights('classifcador_breast.h5')
