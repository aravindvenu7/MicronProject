import timeit
from timeit import default_timer as timer
from torchvision.utils import save_image
import torch
metrics = []
class Test:


    def __init__(self, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, iteration_loss=False):
       
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()
        
        
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            start1 = timer()
            inputs = batch_data[0]#.to(self.device)
            end1 = timer()
            labels = batch_data[1]#.to(self.device)

            with torch.no_grad():
                # Forward propagation
                start2 = timer()
                outputs = self.model(inputs)
                end2 = timer()
                print("each batch takes:")
                print(end1 + end2 - start1 - start2)
                # Loss computation
                loss = self.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of evaluation the metric
            self.metric.add(outputs.detach(), labels.detach())
            metrics.append(self.metric.value())
            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()