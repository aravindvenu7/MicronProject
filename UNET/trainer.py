from keras.optimizers import SGD, RMSprop ,Adam
from networks import make_unet
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint,EarlyStopping
from utils import save_all, plot_metrics
from keras.utils import plot_model
from metrics import soft_dice_loss, dice_coeff, sp, sn
BATCH_SIZE = 3
seed = 1
img_shape = (224, 288, 3)
mask_shape = (224, 288, 1)

def run_modelpat(train_generatorp, test_generatorp, val_generatorp):

  adam = Adam(lr = 3e-5)
  #randomly shuffling training data
  #X_train , y_train = shuffle(X_train , y_train)
  scores = []
 
  print("current running model is model pat ")
  model = make_unet(img_shape)
  #model.load_weights('/tmp/weights25th_logging'+str(i)+'.hdf5')
  model.compile(optimizer = adam , loss = soft_dice_loss , metrics = [dice_coeff, 'acc' , sp , sn])
  model.summary()
  save_model_path = 'Other/unetartery.hdf5'
  earlystopper = EarlyStopping(patience=15, verbose=1)
  cp = ModelCheckpoint(filepath=save_model_path, monitor='val_loss', save_best_only=True, verbose=1)

  history = model.fit_generator(train_generatorp, 
                   steps_per_epoch= 90,
                   epochs=200,
                   validation_data=val_generatorp,
                   validation_steps=25,
                    callbacks=[cp,earlystopper])# , #earlystopper])#,tensorboard_callback])
  scores.append(model.evaluate_generator(test_generatorp,steps =  12,workers = 1,verbose=1))  
  save_all(model,25)
  plot_metrics(history,25)
  print("finished model ")
  #model.reset_states()
  return scores,model  