import torch
import numpy as np
import torch.nn as nn
from networks import FastSCNN
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import transforms as ext_transforms
from metrics import Metric, IoU, ConfusionMatrix
from helper import get_filenames, pil_loader, custom_loader, binarize_array, leave_one_out_datasets, batch_transform, imshow_batch, save_checkpoint, load_checkpoint, load_dataset 
from dataset import Dataset
from networks import FastSCNN, _ConvBNReLU, _DSConv, _DWConv, LinearBottleneck, PyramidPooling, LearningToDownsample, GlobalFeatureExtractor, FeatureFusionModule, Classifier, Interpolate
from train import Train
from test import Test 
#PATHREV is the path to the root directory
save_dir = pathrev + "\outputs"
device = torch.cuda.get_device_name
height = 224
width = 288
batch_size = 8
epochs = 20

path = #give path
learning_rate = 5e-4
lr_decay = 0.1
lr_decay_epochs = 100
weight_decay = 2e-4
path = pathrev
workers = 4
name = 'FSCNN'
printstep = "True"
name = 'FastSCNNartery'
save_dir = pathrev + "\outputs"
imshowbatch = True

def train(train_loader, val_loader, class_weights,class_encoding):
    print("\nTraining...\n")
    num_classes = len(class_encoding)

    resume = False
    
    #resume = True
    model = FastSCNN(num_classes)#.cuda()#.to(device)
    #summary(model, input_size=(3, 192, 192))
    #print(model)


    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()

    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, lr_decay_epochs,
                                     lr_decay)

    # Evaluation metric
    ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Optionally resume from a checkpoint
    if resume:
        model, optimizer, start_epoch, best_miou = load_checkpoint(model, optimizer, '/content/drive/My Drive/originals/outputs',
                          'FastSCNNartery')
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, metric, device)
    val = Test(model, val_loader, criterion, metric, device)
    #val = Test(model, val_loader, criterion, metric, device)
    for epoch in range(start_epoch, epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        lr_updater.step()
        #epoch_loss, (iou, miou) = train.run_epoch(printstep)
        epoch_loss, (iou, miou,acc,sp,sn) = train.run_epoch(printstep)               #,acc,sp,sn)

        #print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f} | Accuracy: {2:.4f}".format(epoch, epoch_loss, miou))
        #print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f} | Accuracy: {3:.4f} | Specificty : {4:.4f} | Sn : {5:.4f}".format(epoch, epoch_loss, miou,acc,sp,sn))
        print("Epoch , loss , iou , acc , sp, sn")
        print(epoch , epoch_loss , miou , np.mean(acc) , sp[1] , sn[1])
        if (epoch % 1 == 0) or epoch + 1 == epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            #loss, (iou, miou) = val.run_epoch(printstep)
            #loss, (iou, miou,acc,sp[1],sn[1]) = val.run_epoch(printstep)
            #print("Epoch , loss , iou , acc , sp, sn")
            #print(epoch , epoch_loss , miou , np.mean(acc) , sp[1] , sn[1])

            #print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, loss, miou))

            # Print per class IoU on last epoch or if best iou
            #if epoch + 1 == epochs or miou > best_miou:
            #    for key, class_iou in zip(class_encoding.keys(), iou):
            #        print("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            #if miou > best_miou:
            print("\nBest model thus far. Saving...\n")
            best_miou = miou
            save_checkpoint(model, optimizer, epoch + 1, best_miou,name,path)
            print("saved checkpoint")
                                      

    return model

def predict(model, images,labels, class_encoding,i, predtimes):
    images = images
    labels = labels
    # Make predictions!
    model.eval()
    with torch.no_grad():
        start = timer()
        predictions = model(images)
        

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)
      ##################################################################################################
    #label_to_rgb = transforms.Compose([ext_transforms.LongTensorToRGBPIL(class_encoding),transforms.ToTensor()])
    
    #imshow_batch(images.data.cpu(),color_labels)
     ####################################################################################################### 
    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = batch_transform(predictions.cpu(), label_to_rgb)
    color_labels = batch_transform(labels, label_to_rgb)
    end = timer()
    predtimes.append(end-start)
    print(end-start)
    #color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu() , color_labels)
    utils.imshow_batch(images.data.cpu(), color_predictions)
    #j = Image.fromarray(color_predictions)
   
    save_image(color_predictions[0],'/content/drive/My Drive/originals/segnew/' +  'left_seg' + str(i).zfill(4) + '.png')
    save_image(images.data.cpu(),'/content/drive/My Drive/originals/segnew/' +  'left' + str(i).zfill(4) + '.png')
    #save_image(labels.data.cpu(),'/content/drive/My Drive/originals/segnew/' +  'leftgt' + str(i).zfill(4) + '.png')
    print(i)
    
    
    #imshow_batch(images.data.cpu(), color_predictions)

def test(model, test_loader, class_weights, class_encoding, predtimes):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    # Evaluation metric
    #if args.ignore_unlabeled:
      #  ignore_index = list(class_encoding).index('unlabeled')
    #else:
    ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")
    start = timer()
    #loss, (iou, miou) = test.run_epoch(printstep)
    loss, (iou, miou,acc,sp,sn) = test.run_epoch(printstep)
    end = timer()
    print("batch time:")
    print(end-start)
    class_iou = dict(zip(class_encoding.keys(), iou))

    #print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))
    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou,acc,sp,sn))
    print("Test Epoch loss , iou , acc , sp, sn")
    print( loss , miou , np.mean(acc) , sp[1] , sn[1])
    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))
    i = 0
    # Show a batch of samples and labels
    for i in range(15):
        
        print("A batch of predictions from the test set...")
        startx = timer()
        images,labels = iter(test_loader).next()
        endx = timer()
        print("loading time:")
        print(endx - startx)
        predict(model, images,labels, class_encoding,i,predtimes)
