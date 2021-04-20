import os, shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def get_filenames(path1):
 ims = []
 masks = []
 for filename in os.listdir(path1):
     
     if ("_" not in filename):
            ims.append(path1 + filename)
            #print("here")
     else:
            masks.append(path1 + filename)
 return ims,masks

def all_together(X,y):
 X_train = []
 y_train = []
 X_test = []
 y_test = []
 X_val = []
 y_val = []
 X,y  =shuffle(X,y)
 X_traint, X_validt, y_traint, y_validt = train_test_split(X[:-36],y[:-36],test_size = 0.2,random_state = 42)
 X_testt = X[-36:]
 y_testt = y[-36:]
 print(X_traint.shape)
 X_train.append(X_traint)
 X_val.append(X_validt)
 y_train.append(y_traint)
 y_val.append(y_validt)
 X_test.append(X_testt)
 y_test.append(y_testt) 

 return X_train,y_train,X_test,y_test,X_val,y_val


def save_all(model,i):
 model_json = model.to_json()
 with open("Other/unetartery" + str(i) + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
 model.save_weights("Other/unetartery" + str(i) +".h5")
 print("Saved model to disk")

def plot_metrics(history,i):
  # Plot training & validation accuracy values
 plt.plot(history.history['acc'])
 plt.plot(history.history['val_acc'])
 plt.title('Model accuracy')
 plt.ylabel('Accuracy')
 plt.xlabel('Epoch')
 plt.legend(['Train', 'Validation'], loc='upper left')
 plt.show()
 #plt.savefig('/accs/acc' + str(i) + '.png')

# Plot training & validation loss values
 plt.plot(history.history['loss'])
 plt.plot(history.history['val_loss'])
 plt.title('Model loss')
 plt.ylabel('Loss')
 plt.xlabel('Epoch')
 plt.legend(['Train', 'Validation'], loc='upper left')
 plt.show()
 #plt.savefig('/losses/loss' + str(i)+'.png')

# Plot training & validation loss values
 plt.plot(history.history['dice_coeff'])
 plt.plot(history.history['val_dice_coeff'])
 plt.title('Model DSC')
 plt.ylabel('DSC')
 plt.xlabel('Epoch')
 plt.legend(['Train', 'Validation'], loc='upper left')
 plt.show()
 #plt.savefig('/DSC/dsc'+str(i)+'.png')
 
 plt.plot(history.history['sn'])
 plt.plot(history.history['val_sn'])
 plt.title('Model Sensitivity')
 plt.ylabel('Sensitivity')
 plt.xlabel('Epoch')
 plt.legend(['Train', 'Validation'], loc='upper left')
 plt.show()
 #plt.savefig('SN/sn'+str(i)+'.png')

 plt.plot(history.history['sp'])
 plt.plot(history.history['val_sp'])
 plt.title('Model Specificity')
 plt.ylabel('Specificity')
 plt.xlabel('Epoch')
 plt.legend(['Train', 'Validation'], loc='upper left')
 plt.show()
 #plt.savefig('SP/sp'+str(i)+'.png')