class Train:
  def  __init__(self,model,data_loader,optim,criterion,metric,device):
    self.model = model
    self.data_loader = data_loader
    self.optim = optim
    self.criterion = criterion
    self.metric = metric
    self.device = device
  def run_epoch(self,iteration_loss = False):
    self.model.train()
    epoch_loss = 0.0
    self.metric.reset()
    i = 0
    for step,batch_data in enumerate(self.data_loader):
      i = i+1
      #if(i%5 == 0 and i != 0):
        #break
      image = batch_data[0]#.to(self.device)
      label = batch_data[1]#.to(self.device)
      #label = label.type(torch.LongTensor)                                                               #,y=x.type(torch.DoubleTensor),y.type(torch.DoubleTensor)
      #print(label.size() , label.dtype)
      outputs = self.model(image)
      #outputs = outputs.type(torch.LongTensor)
      #p#rint(outputs.size() , outputs.dtype)
      loss = self.criterion(outputs,label)
      self.optim.zero_grad()
      loss.backward()
      self.optim.step()
      epoch_loss = epoch_loss + loss.item()
      self.metric.add(outputs.detach(),label.detach())
      if iteration_loss:
        print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
        
    return epoch_loss/ len(self.data_loader) , self.metric.value()
