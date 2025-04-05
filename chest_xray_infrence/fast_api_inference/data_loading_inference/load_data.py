import cv2
import numpy as np
from PIL import Image
import io

def load(file):

    if file.content_type.split(sep = '/')[0] == 'image':

        image = file.file.read()
        image = np.array(Image.open(io.BytesIO(image)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, interpolation= cv2.INTER_CUBIC, dsize = (224,224)).reshape((1,224,224,3))
        return image
        

    else:

        return None