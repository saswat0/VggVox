import os
import torch
import numpy as np
import tensorboardX
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from network import VggVox
from dataset import VoxLoader
from utils import Normalize, ToTensor
from torchvision.transforms import Compose

DATASET_PATH = '/home/saswat/datasets/wav/'
LOG_DIR = '/home/saswat/datasets/logs'
total_epochs = 30

torch.backends.cudnn.deterministic = True
batch_size = 96

decay = 5e-4
initial_lr = 1e-2
final_lr = 1e-4

gamma = 10 ** (np.log10(final_lr / initial_lr) / (total_epochs - 1))
device = 'cuda:0'

TBoard = tensorboardX.SummaryWriter(log_dir=LOG_DIR)

model = VggVox(num_classes=1251)
model.to(device)

transforms = Compose([
    Normalize(),
    ToTensor()
])

trainset = VoxLoader(DATASET_PATH, train=True, transform=transforms)
trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True)

testset = VoxLoader(DATASET_PATH, train=False, transform=transforms)
testsetloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=8)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), initial_lr, 0.9, weight_decay=decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

# Training
for epoch_num in range(total_epochs):
    lr_scheduler.step()
    
    model.train()
    
    for iter_num, (labels, specs) in enumerate(trainsetloader):
        optimizer.zero_grad()
        labels, specs = labels.to(device), specs.to(device)
        scores = model(specs)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        
        # Logging
        step_num = epoch_num * len(trainsetloader) + iter_num
        TBoard.add_scalar('Metrics/train_loss', loss.item(), step_num)
        TBoard.add_scalar('Metrics/lr', lr_scheduler.get_lr()[0], step_num)
            
    # Evaluation
    model.eval()
    
    top5_accuracy = 0
    top1_accuracy = 0

    for _, (label, spec) in tqdm(enumerate(testsetloader)):
        label, spec = label.to(device), spec.to(device)
        probs = model(spec)

        # calculate Top-5 and Top-1 accuracy
        pred_top5 = probs.topk(5)[1].view(5)

        if label in pred_top5:
            # increment top-5 accuracy
            top5_accuracy += 1

            if label == pred_top5[0]:
                # increment top-1 accuracy
                top1_accuracy += 1

    top5_accuracy /= len(testsetloader)
    top1_accuracy /= len(testsetloader)

    TBoard.add_scalar('Metrics/test_top5', top5_accuracy, epoch_num)
    TBoard.add_scalar('Metrics/test_top1', top1_accuracy, epoch_num)

torch.save(model.state_dict(), os.path.join(LOG_DIR, 'checkpoint.txt'))
TBoard.close()
print('Top 1 accuracy: {}'.format(round(top1_accuracy, 3)))
print('Top 5 accuracy: {}'.format(round(top5_accuracy, 3)))