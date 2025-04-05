import numpy as np
import cv2
from keras.applications.resnet import preprocess_input

# Local Modules
from configs import configs



def nn_train_gen(X_train, y_train, batch_size = configs.BATCH_SIZE):

    '''Return a generator based on the X_train, y_train and the batch_size.
    the generator will return a tuple of preprocessed image with the labels'''

    shuffled_index = np.random.permutation(range(X_train.shape[0]))
    pointer_location = 0

    y_train = y_train.to_numpy()
    
    X_mini_batch_shape = (batch_size, 224, 224, 3)
    X_mini_batch = np.zeros(shape = X_mini_batch_shape, dtype = np.float64)
    y_mini_batch = np.zeros(shape = (batch_size, 14),dtype = int)

    while True:

        for j in range(batch_size):
            
            img = cv2.imread(X_train['images'][shuffled_index[pointer_location]])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,configs.IMAGE_RESIZE_DIM , interpolation = cv2.INTER_CUBIC)

            X_mini_batch[j] = img
            y_mini_batch[j] = y_train[shuffled_index[pointer_location]]
            
            pointer_location = pointer_location + 1

        if (abs(pointer_location - shuffled_index.shape[0]) < batch_size):
            shuffled_index = np.random.permutation(range(X_train.shape[0]))
            pointer_location = 0
        
        yield (preprocess_input(X_mini_batch), y_mini_batch)



