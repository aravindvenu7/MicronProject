import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, RepeatVector, Permute, Reshape,concatenate,LeakyReLU,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D ,Conv2DTranspose,Add,Concatenate
from keras.layers import LeakyReLU, BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, RMSprop ,Adam
from keras.utils.layer_utils import print_summary
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint,EarlyStopping
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def conv(input,num_filters):
  layer1 = Conv2D(num_filters , (3,3) , padding = 'same')(input)
  layer2  = BatchNormalization()(layer1)
  layer3 = Activation('relu')(layer2)
  layer3 = Dropout(0.25)(layer3)
  layer4 = Conv2D(num_filters , (3,3) , padding = 'same')(layer3)
  layer5  = BatchNormalization()(layer4)
  layer6 = Activation('relu')(layer5)
  return layer6

def encoder(input,num_filters):
  x = conv(input,num_filters)
  pool = MaxPooling2D((2,2), strides = (2,2))(x)
  return pool , x



def decoder(input,concat,num_filters):
   x = Conv2DTranspose(num_filters , (2,2) ,strides = (2,2) , padding = 'same')(input)
   x = concatenate([x,concat] , axis = -1)
   x = BatchNormalization()(x)
   decoder = Conv2D(num_filters, (3, 3), padding='same')(x)
   decoder = BatchNormalization()(decoder)
   decoder = Activation('relu')(decoder)
   decoder = Dropout(0.25)(decoder)
   decoder = Conv2D(num_filters, (3, 3), padding='same')(decoder)
   decoder = BatchNormalization()(decoder)
   decoder = Activation('relu')(decoder)
   return decoder



def make_unet(image_shape):

 inputs  = Input(shape = image_shape)
 #inputs = Lambda(lambda x: x / 255) (inputs)
 #enp1 , en1 = encoder(inputs , 32)
 enp2 , en2 = encoder(inputs , 64)
 enp3 , en3 = encoder(enp2 , 128)
 enp4 , en4 = encoder(enp3 , 256)
 #enp5 , en5 = encoder(enp4 , 512)
 center = conv(enp4 , 512)
 decoder4 = decoder(center, en4, 256)
 decoder3 = decoder(decoder4, en3,128)
 decoder2 = decoder(decoder3, en2, 64)
 #decoder1 = decoder(decoder2, en2, 64)
 #decoder0 = decoder(decoder1, en1, 32)
 outputs = Conv2D(1 ,(1,1) , activation = 'sigmoid')(decoder2)  
 mod = Model(inputs = [inputs] , outputs = [outputs])
 return mod