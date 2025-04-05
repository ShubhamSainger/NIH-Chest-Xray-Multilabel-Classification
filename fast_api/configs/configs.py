from training.metrics import precision, recall

ROOT_DIR_PATH = '.'


DISEASE_ENCODINGS_DIR = ROOT_DIR_PATH + '/encoding'
DISEASE_ENCODINGS_FILE_NAME = 'diseases_encodings.pkl'
DISEASE_ENCODINGS_PATH =ROOT_DIR_PATH +  '/encoding/diseases_encodings.pkl'
DISEASES = ['Atelectasis', 'Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration',
            'Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']


DATA_DIR = ROOT_DIR_PATH + '/dataset'
DATA_FILE_NAME = 'data.csv'
DATA_CSV_PATH = ROOT_DIR_PATH + '/dataset/csv/data.csv'
DATA_IMAGE_PATH = ROOT_DIR_PATH + '/dataset/images/'


MODEL_WEIGHTS_DIR = ROOT_DIR_PATH + '/model'
MODEL_WEIGHTS_FILE = 'Model.weights.h5'
MODEL_WEIGHTS_PATH = ROOT_DIR_PATH + '/model/Model.weights.h5'


TRAINING_DATA_SIZE = 0.7
IS_SPLIT = False
BATCH_SIZE = 2
IMAGE_RESIZE_DIM = (224,224)
DECISION_THRESHOLD = 0.7

METRICS = [precision, recall]
OPTIMIZER = 'adam'
EPOCHS = 1

IS_RETRAINING = False