import pickle
import numpy as np
from configs import configs

with open(configs.DISEASE_ENCODINGS_PATH, mode = 'rb') as f:
    diseases_key_value = pickle.load(f)

def multi_hot_encoder(x):
    one_hot_encoded_vector = np.eye(len(configs.DISEASES))
    temp = np.zeros((1,len(configs.DISEASES)))
    for i in x:
        if i not in configs.DISEASES:      # If i is not in diseases we are considering as normal
            return np.zeros((1,14))
        temp[0] = temp[0] + one_hot_encoded_vector[diseases_key_value[i]]
    return temp
