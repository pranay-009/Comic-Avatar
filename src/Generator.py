import keras 
import tensorflow as tf
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.initializers import RandomNormal
from keras.layers import concatenate,MaxPooling2D



def Generator(x,y,z):
    inputs=Input(shape=(x,y,z))
    init = RandomNormal(stddev=0.02)
    c1=Conv2D(64,(4,4),strides=(2,2),activation="relu",kernel_initializer=init,padding="same")(inputs)
    c1=LeakyReLU(alpha=0.2)(c1)#(64,64,64)

    c2 = Conv2D(128, (4,4),strides=(2,2), kernel_initializer=init, padding='same')(c1)
    c2=BatchNormalization()(c2,training=True)#(128,32,32)
    c2=LeakyReLU(alpha=0.2)(c2)

    c3 = Conv2D(256 ,(4,4), strides=(2,2) ,kernel_initializer=init, padding='same')(c2)
    c3=BatchNormalization()(c3,training=True)#(256,16,16)
    c3=LeakyReLU(alpha=0.2)(c3)

    c4 = Conv2D(512, (4,4), strides=(2,2), kernel_initializer=init, padding='same')(c3)
    c4=BatchNormalization()(c4,training=True)#(512,8,8)
    c4=LeakyReLU(alpha=0.2)(c4)

    c5 = Conv2D(512, (4,4),strides=(2,2),  kernel_initializer=init, padding='same')(c4)
    c5=BatchNormalization()(c5,training=True)#512,4,4
    c5=LeakyReLU(alpha=0.2)(c5)  

    c6 = Conv2D(512, (4,4), strides=(2,2),kernel_initializer=init, padding='same')(c5)
    c6=BatchNormalization()(c6,training=True)#512,2,2
    c6=LeakyReLU(alpha=0.2)(c6)  

    c7 = Conv2D(512, (4,4), strides=(2,2), kernel_initializer=init, padding='same')(c6)

    b = Activation('relu')(c7)  

    d1 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(b)
    d1=BatchNormalization()(d1,training=True)#512*2*2
    d1=Dropout(0.5)(d1,training=True)
    d1=Concatenate()([d1, c6])   
    d1=Activation("relu")(d1)

    d2 =Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d1)
    d2=BatchNormalization()(d2,training=True)#512*4*4
    d2=Dropout(0.5)(d2,training=True)
    d2=Concatenate()([d2,c5])  
    d2=Activation("relu")(d2)

    d3 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d2)
    d3=BatchNormalization()(d3,training=True)#512*8*8
    d3=Dropout(0.5)(d3,training=True)
    d3=Concatenate()([d3, c4])   
    d3=Activation("relu")(d3)

    d4 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d3)
    d4=BatchNormalization()(d4,training=True)#512*16*16
    d4=Concatenate()([d4, c3])   
    d4=Activation("relu")(d4) 

    d5 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d4)
    d5=BatchNormalization()(d5,training=True)#256*32*32
    d5=Concatenate()([d5, c2])   
    d5=Activation("relu")(d5) 

    d6 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
    d6=BatchNormalization()(d6,training=True)#128*64*64
    d6=Concatenate()([d6, c1])   
    d6=Activation("relu")(d6)     

    f = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d6) 
    out_image = Activation('tanh')(f) #3,128,128

    model = Model(inputs, out_image)
    return model
