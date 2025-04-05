import sys
sys.path.append("/app")
from configs import configs 
from training import data_loading, weighted_loss
from data_cleaning import data_cleaning
from model import model

import sys
sys.path.append("/home/kakashi/Deployement-Project/Chest-Xray/fast_api")

def start_training():

    X, y = data_cleaning.data_cleaner()

    data_generator = data_loading.nn_train_gen(X, y)
    training_steps = X.shape[0]//configs.BATCH_SIZE
    

    network = model.network()

    network.compile(optimizer = configs.OPTIMIZER, loss = weighted_loss.Weighted_Binary_CLE, metrics = configs.METRICS)

    network.fit(data_generator, epochs = configs.EPOCHS, steps_per_epoch = training_steps)

start_training()
