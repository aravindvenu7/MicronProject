import numpy as np
import cv2, torch
import torchvision
from skimage.morphology import skeletonize
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.utils.data as data
import torchvision.transforms as transforms
BATCH_SIZE = 8
seed = 1
img_shape = (224, 288, 3)
mask_shape = (224, 288, 1)
batch_size = 8
im = 224
ma = 224
workers = 4
imshowbatch = True


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



def pil_loader(data_path, label_path):
   
    data = Image.open(data_path)
    #data = cv2.resize(data,(192,192))
    
    label = Image.open(label_path)

    label = label.convert('L')  # convert image to monochrome
    label = np.array(label)
    label = cv2.resize(label,(224,288))
    label = binarize_array(label, threshold=50)

    #label = skeletonize(label)
    #print(label.shape)
    #label = label.astype(np.uint8)
    label = Image.fromarray(label)

    return data, label

def custom_loader(data_path):
  data = Image.open(data_path)
  return data

def binarize_array(numpy_array, threshold=50):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 1
            else:
                numpy_array[i][j] = 0
    return numpy_array


#loop for generating all 9 datasets:
def leave_one_out_datasets(X,y):
  X_train = []
  y_train = []
  X_test = []
  y_test = []
  X_val = []
  y_val = []
  #for i in range(8):
  i=2
  print(240*(i+1))

  
  X_test = X[-36:]
  y_test = y[-36:]
  X_train = X[:-72]
  X_val = X[-72:-36]
  y_train = y[:-72]
  y_val = y[-72:-36]
  
  
  return X_train,y_train,X_test,y_test,X_val,y_val


def batch_transform(batch,transform):
  transf_slices = [transform(tensor ) for tensor in torch.unbind(batch)]
  return torch.stack(transf_slices)

def imshow_batch(images, labels):

    

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()
    #gts = torchvision.utils.make_grid(gts).numpy()


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def save_checkpoint(model,optimizer,epoch,miou,name, path):
  save_dir = path + "/outputs"
  model_path = save_dir + "/" + name
  checkpoint = {
    'epoch': epoch,
    'miou': miou,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()}
    
  torch.save(checkpoint, model_path)

  summary_file = save_dir + "/" + "FastSCNNartery.txt"
  with open(summary_file,'w') as file:
        file.write("\nBEST VALIDATION\n")
        file.write("Epoch: {0}\n". format(epoch))
        file.write("Mean IoU: {0}\n". format(miou))

def load_checkpoint(model, optimizer, folder_dir, filename):
   
  
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)
    model_path = folder_dir + "/" + "FastSCNNartery"
    # Create folder to save model and information
    #model_path = os.path.join(folder_dir, filename)
    #assert os.path.isfile(
    #    model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    return model, optimizer, epoch, miou


def load_dataset(dataset,mode, X_train, y_train, X_test, y_test, X_val, y_val):
  print("Loading Dataset...")
  image_transform = transforms.Compose([transforms.Resize((height, width)),transforms.ToTensor(),])
  label_transform = transforms.Compose([transforms.Resize((height, width), Image.NEAREST),ext_transforms.PILToLongTensor()])
  train_set = dataset(mode = 'train', transform = image_transform,label_transform = label_transform, X_train, y_train, X_test, y_test, X_val, y_val)
  train_loader = data.DataLoader(train_set,batch_size,shuffle = True,num_workers = workers)
  
  val_set = dataset(mode = 'val', transform = image_transform,label_transform = label_transform,  X_train, y_train, X_test, y_test, X_val, y_val)
  val_loader = data.DataLoader(val_set,batch_size,shuffle = True,num_workers = workers)
  print(val_set)
  
  test_set = dataset(mode = 'test', transform = image_transform,label_transform = label_transform, X_train, y_train, X_test, y_test, X_val, y_val)
  print(test_set)
  test_loader = data.DataLoader(test_set,batch_size,shuffle = True,num_workers = workers)
  
  class_encoding = train_set.color_encoding
  num_classes = len(class_encoding)
  
  print("Number of classes to predict:",len(class_encoding))
  print("Train dataset size:", len(train_set))
  print("Validation dataset size:", len(val_set))
  if mode.lower() == 'test':
      images,labels = iter(test_loader).next()
      print("We are currently here")
      #print("Image size:", images.size())
  else:
      images, labels = iter(train_loader).next()
  #print("Image size:", images.size())
  #print("Label size:", labels.size())
  print("Class-color encoding:", class_encoding)
  #numpy_labels = labels.numpy()
  #print(np.unique(numpy_labels))
  if imshowbatch:
    print("Close the figure window to continue...")
    
    label_to_rgb = transforms.Compose([ext_transforms.LongTensorToRGBPIL(class_encoding),transforms.ToTensor()])
    color_labels = batch_transform(labels, label_to_rgb)
    imshow_batch(images,color_labels) # color_labels)
    #print(color_labels.size())
    print("here")
    
  #class_weights = enet_weighing(train_loader, num_classes)
  #class_weights = torch.from_numpy(class_weights).float()#.to(device)
  print("also here")
  class_weights = 1
  return (train_loader, val_loader,test_loader), class_weights,class_encoding
