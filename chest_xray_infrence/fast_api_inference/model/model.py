from keras.applications import VGG16, VGG19, ResNet50, ResNet101
from keras.layers import Input, Concatenate, GlobalAveragePooling2D, BatchNormalization, Dense
from keras.models import Model

def network():

    input_vector = Input(shape = (224,224,3))

    vgg_16 = VGG16(include_top = False, weights = None)
    vgg_19 = VGG19(include_top = False, weights = None)
    resnet_50 = ResNet50(include_top = False, weights = None)
    resnet_101 = ResNet101(include_top = False, weights = None)

    vgg_16.trainable,vgg_19.trainable,resnet_50.trainable, resnet_101.trainable = False, False, False, False

    x_1 = vgg_16(input_vector)
    x_2 = vgg_19(input_vector)
    x_3 = resnet_50(input_vector)
    x_4 = resnet_101(input_vector)
    

    x = Concatenate()([x_1,x_2,x_3,x_4])

    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(units = 256, activation = 'relu')(x)
    x = BatchNormalization()(x)
    output_vector = Dense(units = 14, activation = 'sigmoid', kernel_initializer = 'glorot_normal', kernel_regularizer = 'l2')(x)
    
    return Model(input_vector, output_vector)


