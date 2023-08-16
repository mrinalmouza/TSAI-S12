from typing import List, Union
import torch
from pytorch_lightning import LightningModule, Trainer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import DataAugmentations as Aug


BATCH_SIZE = 1


class LIT_Cifar10(LightningModule):
    def __init__(self):
        super(LIT_Cifar10, self).__init__()
        #Input image = [B, 3, 32, 32]
        #Input image = [B, 3, 32, 32]
        self.prep_layer     = nn.Sequential(
                                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding= 1, stride=1, bias = False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        #nn.Dropout(.02),
                                        )
        #Output of prep layer = [B, 64, 32, 32], Receptive Field = 3
        self.layer1         = nn.Sequential(
                                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, bias = False),
                                        nn.MaxPool2d(2,2),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        # nn.Dropout(.04),    
                                        )
        #Output of layer1 = [B, 128, 16, 16], Receptive field = 6
        self.residual_block1 = nn.Sequential(
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,  padding =1, stride=1, bias = False),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        #nn.Dropout(.01),
                                        nn.Conv2d(in_channels=128, out_channels= 128, kernel_size = 3, padding=1, stride=1, bias = False),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        # nn.Dropout(.01),    
                                        )
        #Output of layer1 = [B, 128, 16, 16], Receptive field = 14
        self.layer2         = nn.Sequential(
                                        #nn.ReLU(),
                                        nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3, padding=1, stride=1, bias = False),
                                        nn.MaxPool2d(2,2),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        # nn.Dropout(.15),

                                        )
        #Output of layer1 = [B, 256, 8, 8], Receptive field = 18
        self.layer3         = nn.Sequential(
                                        nn.Conv2d(in_channels=256, out_channels=512,kernel_size=3, stride=1, padding=1, bias = False),
                                        nn.MaxPool2d(2,2),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        # nn.Dropout(.20),
                                        )     
        #Output of layer1 = [B, 512, 4, 4], Receptive field = 26
        self.residual_block2 = nn.Sequential(
                                        nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3, padding =1, stride=1, bias = False),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                       # nn.Dropout(.01),
                                        nn.Conv2d(in_channels=512, out_channels= 512,kernel_size=3, padding=1, stride=1, bias = False),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        # nn.Dropout(.01),    
                                        )
       

        self.maxpool = nn.MaxPool2d(4, 4)

        #Output of layer1 = [B, 512, 1, 1]

        self.fc1 = nn.Linear(512, 10, bias = False)

    def forward(self, x):
        
        x = self.prep_layer(x)
        x = self.layer1(x)
        
        r1 = self.residual_block1(x)
        x1 = x + r1
        
        x = self.layer2(x1)
        x = self.layer3(x)
        r2 = self.residual_block2(x)
        x2 = x+ r2
        
        x = self.maxpool(x2)
        x = x.view(-1, 512)

      
        
        x = self.fc1(x)
        

        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= 0.01, weight_decay= 1e-4)

        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        tensorboard_logs = {'train_loss': loss}
        return loss, tensorboard_logs
    
    def train_dataloader(self):
        train_data = Aug.Cifar10Dataset(root='../data', train=True, download=False, transform=Aug.train_transforms)
        train_loader = DataLoader(dataset = train_data, batch_size  = BATCH_SIZE, shuffle = True)
        return train_loader
    
    def test_dataloader(self):
        test_data = Aug.Cifar10Dataset(root='../data', train=False,download=False, transform=Aug.test_transform)
        test_loader = DataLoader(dataset = test_data, batch_size  = BATCH_SIZE, shuffle = True)
        return test_loader
    
trainer = Trainer(fast_dev_run = True,accelerator="mps")
model = LIT_Cifar10()
trainer.fit(model)
