import torch
import torch.utils.data as data
from collections import OrderedDict
from helper import pil_loader, custom_loader
class Dataset(data.Dataset):
  color_encoding = OrderedDict([('vessel',255),('nonvessel',0)])
  
  def __init__(self,mode = 'train', transform = None,label_transform = None,loader = [pil_loader,custom_loader], X_train, y_train, X_test, y_test, X_val, y_val):
    self.mode = mode
    self.loader = loader
    self.transform = transform
    self.label_transform = label_transform
    
    
    if self.mode.lower() == 'train' :
      self.train_data = X_train
      self.train_labels = y_train
    if self.mode.lower() == 'val' : 
      self.val_data = X_val
      self.val_labels = y_val
      
    if self.mode.lower() == 'test'  :
      self.test_data = X_test
      self.test_labels = y_test
      

  def __getitem__(self,index):
    
   if self.mode.lower() == 'train' :
    img,label =  self.loader[0](X_train[index],y_train[index])
      
   if self.mode.lower() == 'val' : 
    img,label = self.loader[0](X_val[index],y_val[index])
   if self.mode.lower() == 'test'  :
    #img = self.loader[1](X_testnew[index])
    
    img ,label =  self.loader[0](X_test[index],y_test[index]) 
  
   if self.transform is not None:
         img = self.transform(img)
 
   if self.label_transform is not None: #and self.mode.lower() != 'test':
         label = self.label_transform(label)
   if self.mode.lower() == 'test':
      return img,label
   else:
      return img,label #.unsqueeze_(0)

  def __len__(self):
       
        if self.mode.lower() == 'train':
            return len(X_train)
        elif self.mode.lower() == 'val':
            return len(X_val)
        elif self.mode.lower() == 'test':
            return len(X_test)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
