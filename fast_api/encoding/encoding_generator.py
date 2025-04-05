## adding the root path
from sys import path
path.append("./fast_api")


## 

import pickle
from configs import configs



diseases_encodings = {'Atelectasis': 0,
 'Cardiomegaly': 1,
 'Consolidation': 2,
 'Edema': 3,
 'Effusion': 4,
 'Emphysema': 5,
 'Fibrosis': 6,
 'Hernia': 7,
 'Infiltration': 8,
 'Mass': 9,
 'Nodule': 10,
 'Pleural_Thickening': 11,
 'Pneumonia': 12,
 'Pneumothorax': 13}

with open(configs.DISEASE_ENCODINGS_PATH, mode='wb') as f:
    pickle.dump(diseases_encodings,f)