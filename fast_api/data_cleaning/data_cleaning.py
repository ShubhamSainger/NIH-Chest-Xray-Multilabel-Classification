import pandas as pd
import numpy as np

## Local Modules
# from sys import path
# print(path)
# path.append("/home/kakashi/Deployement-Project/Chest-Xray/fast_api")
from configs import configs
from data_cleaning.multi_hot_encoder import multi_hot_encoder



def data_cleaner(path = configs.DATA_CSV_PATH):

    '''path : 
                path of the .csv file which contains the names and labels of the images 

    return: 
            X and y if config.IS_SPLIT = False
            if config.IS_SPLIT = True then it will return X_train, y_train, X_test, y_test'''

    data = pd.read_csv(path)
    data = pd.DataFrame({'images' : data['Image Index'].values,'labels':data['Finding Labels']})
    data = data.dropna(axis = 0)
    path = configs.DATA_IMAGE_PATH
    data['images'] = path + data['images']
    

    multi_hot_encoded_vector = data['labels'].apply(lambda x : multi_hot_encoder(x.split('|')))
    multi_hot_encoded_vector = np.concatenate(multi_hot_encoded_vector.values).astype(np.int32)

    data = pd.concat([data,pd.DataFrame(multi_hot_encoded_vector)], axis = 1)
    data = data.sample(data.shape[0], ignore_index = True)

    X,y = data.iloc[:,:2], data.iloc[:,2:]

    if configs.IS_SPLIT:
        training_size = int(data.shape[0] * configs.TRAINING_DATA_SIZE)

        X_train, y_train = X.iloc[:training_size], y.iloc[:training_size]
        X_test, y_test = X.iloc[training_size:], y.iloc[training_size:]

        return X_train, y_train, X_test, y_test

    return X,y
