import keras
from keras.preprocessing import image
BATCH_SIZE = 3
seed = 1

def generators(X_train, X_test, X_val, y_train, y_test, y_val):
   

  image_datagen = image.ImageDataGenerator()#rotation_range = 90,width_shift_range = 0.2,zoom_range =0.2,height_shift_range = 0.2,horizontal_flip = True)#,width_shift_range=0.2)#tal_flip=True)
  mask_datagen = image.ImageDataGenerator()#rotation_range = 90,width_shift_range = 0.2,zoom_range =0.2,height_shift_range = 0.2,horizontal_flip = True)
  #Training data is now augmented with a rotation angle of 45 degrees
  image_datagen.fit(X_train[0], augment=False, seed=seed)
  mask_datagen.fit(y_train[0], augment=False, seed=seed)
  x_genp = image_datagen.flow(X_train[0],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
  y_genp = mask_datagen.flow(y_train[0],batch_size=BATCH_SIZE,shuffle=True, seed=seed)

  image_datagen_val = image.ImageDataGenerator()#rotation_range = 90,width_shift_range = 0.2)
  mask_datagen_val = image.ImageDataGenerator()#rotation_range = 90,width_shift_range = 0.2)
  image_datagen_val.fit(X_val[0], augment=False, seed=seed)
  mask_datagen_val.fit(y_val[0], augment=False, seed=seed)
  x_valid_genp = image_datagen_val.flow(X_val[0],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
  y_valid_genp = mask_datagen_val.flow(y_val[0],batch_size=BATCH_SIZE,shuffle=True, seed=seed) 

  image_datagen_test = image.ImageDataGenerator()#rotation_range = 90)
  mask_datagen_test = image.ImageDataGenerator()#rotation_range = 90)
  image_datagen_test.fit(X_test[0], augment=False, seed=seed)
  mask_datagen_test.fit(y_test[0], augment=False, seed=seed)
  x_test_genp = image_datagen_test.flow(X_test[0],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
  y_test_genp = mask_datagen_test.flow(y_test[0],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
  print("done")
  return x_genp,y_genp,x_valid_genp,y_valid_genp,x_test_genp,y_test_genp

#to combine both image and mask generators
def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())

def combinepat(x_genp, x_valid_genp, x_test_genp, y_genp, y_valid_genp, y_test_genp):
  
  train_generatorp = combine_generator(x_genp,y_genp)
  val_generatorp = combine_generator(x_valid_genp,y_valid_genp)
  test_generatorp = combine_generator(x_test_genp,y_test_genp)
  return train_generatorp,val_generatorp,test_generatorp
