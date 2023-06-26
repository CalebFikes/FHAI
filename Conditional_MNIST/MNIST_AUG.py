import time
start_time = time.time()

import copy
import logging
import random
import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sm
from sklearn.metrics import f1_score
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from tqdm.notebook import tqdm
from imblearn.over_sampling import SMOTE

#import torch.utils.tensorboard
#from torch.utils.tensorboard import SummaryWriter

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
torch.manual_seed(8675309)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configs
configs = {
'n_epochs' : 30, 
'batch_size_train' : 128, 
'batch_size_test' : 500, 
'learning_rate' : 0.01, 
'momentum' : 0.2, 
'log_interval' : 10,
'class_labels' : np.array([2,7])
}

configs_DDPM = {
    'n_epoch' : 30,
    "batch_size" : 256, 
    'n_T' : 200, 
    'device' : "cuda:0",
    'n_classes' : 2, 
    'n_feat' : 256, 
    'lrate' : 1e-4,
    '1' : .1
}

# Load datasets from torchvision datasets
train=torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

test=torchvision.datasets.MNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))


# Set up tensorboard
#%pip install tensorboard
#%reload_ext tensorboard
#%reload_ext tensorboard

#%rm -rf ./logs/ # clear tensorboard logs --  COMMENT THIS OUT IN THE FUTURE

# create instance of writer for tensorboard
#writer = SummaryWriter('logs/MNIST_imbalanced') # path to directory containing run data


class PrepareData:
    def __init__(self, train_set, test_set, prop_keep = 1):
        """
        Arguments:
            train_set (torch dataset object)
            test_set (torch dataset object):
        Subsets data to select only desired classes, then imbalances training set, then refactors labels.
        Returns 4 float tensors
        """
        self.train_data, self.train_targets = self.prepare_imbalanced_dataset(train_set, prop_keep)
        self.test_data, self.test_targets = self.prepare_test_dataset(test_set)

    def prepare_test_dataset(self, dataset):
        data, targets = dataset.data, dataset.targets
        data, targets = self.subset_data(data, targets)
        targets = self.refactor_labels(targets)
        return data.float(), targets.float()

    def prepare_imbalanced_dataset(self, dataset, prop_keep):
        data, targets = dataset.data, dataset.targets
        data, targets = self.subset_data(data, targets)
        data, targets = self.imbalance_data(data, targets, prop_keep)
        targets = self.refactor_labels(targets)
        return data.float(), targets.float()

    def subset_data(self, data, targets):
        selection = torch.logical_or(targets == 2, targets == 7)
        data = data[selection]
        targets = targets[selection]
        return data, targets

    def imbalance_data(self, data, targets, prop_keep):
        sample_probs = {'2': (1 - prop_keep), '7': 0}
        idx_to_del = [i for i, label in enumerate(targets) if random.random() > sample_probs[str(label.item())]]
        data = data[idx_to_del]
        targets = targets[idx_to_del].type(torch.float)
        return data, targets

    def refactor_labels(self, targets):
        targets[targets == 2.] = 0
        targets[targets == 7.] = 1
        return targets

def imbalance_data(train,test,prop_keep = 1):
    # Modify the data
    data_preparer = PrepareData(train, test) #, 0.1)
    train.data = data_preparer.train_data
    train.targets = data_preparer.train_targets
    test.data = data_preparer.test_data
    test.targets = data_preparer.test_targets
    return train, test


# Define simple CNN to classify dataset examples
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def vis(train_loss, test_accs, confusion_mtxes, labels, figsize=(7, 5)):
    cm = confusion_mtxes[np.argmax(test_accs)] # select the best run (highest test accuracy); cm is the array of raw counts for confusion matrix
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100 # cm_perc is the values for the confusion matrix
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'

    fig = plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.semilogy(train_loss, 'r')
    plt.ylabel('Log training loss')

    plt.subplot(1, 3, 2)
    plt.title('Test Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('% accurate')
    plt.plot(test_accs, 'g')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    sns.heatmap(cm_df, annot=annot, fmt='', cmap="Blues")
    plt.show()
    return fig

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, label= 2, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise

        c_i = torch.tensor(label).to(device)
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
    
def train_classifier(train, test, configs):
    torch.backends.cudnn.enabled = False

    # Define train loader and test loader
    train_loader = torch.utils.data.DataLoader(train, batch_size=configs['batch_size_train'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=configs['batch_size_test'], shuffle=True)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    model = Net().to(device) # creating an instance of Net() and pushing it to GPU
    optimizer = torch.optim.SGD(model.parameters(), configs['learning_rate'], configs['momentum']) # (optimizer args specified in configs)
    
    train_loss = []
    auroc_list, precision_list, recall_list = [], [], []
    pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1, task = 'binary')
    auroc_metric = torchmetrics.classification.BinaryAUROC(thresholds=None)
    test_accs, confusion_mtxes = [], []
    for epoch in range(1, configs['n_epochs']):
        model.train()
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader, position=0, leave=True)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device) # since I'm using CPU, I do not push these tensors to device 
            optimizer.zero_grad()
            output = model(data).to(device)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(CE=loss.item())

        model.eval()
        correct = 0 # count correct predictions
        train_loss.append(loss.item())
        #writer.add_scalar('Training loss',
        #                       loss.item(),
        #                        epoch)
        targets, preds = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device) # since I'm using CPU, I do not push these tensors to device 
                output = model(data)
                _, pred = torch.max(output,dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()

                targets += list(target.to("cpu").numpy())
                preds += list(pred.to("cpu").numpy())

        test_acc = 100. * correct / len(test_loader.dataset)
        #writer.add_scalar('Test Accuracy', test_acc, epoch)
        confusion_mtx = sm.confusion_matrix(targets, preds)
        confusion_mtxes.append(confusion_mtx)
        test_accs.append(test_acc)
        auroc = auroc_metric(torch.Tensor(preds), torch.Tensor(targets))
        auroc_list.append(auroc)
        print(epoch)
    print(f'\rBest test acc {max(test_accs)}', end='', flush=True)


    # Calculate AUROC, f1, precision, recall
    f1, recall, precision, auroc = f1_score(targets, preds, average='macro'), sm.recall_score(targets, preds), sm.precision_score(targets, preds), sm.roc_auc_score(targets, preds)

    print(f'f1 score: {f1} \n recall: {recall} \n precision: {precision} \n Area under receiving operating characteristic: {auroc}')

    #writer.add_figure('matplotlib', vis(train_loss, test_accs, confusion_mtxes, configs['class_labels'], figsize=(15, 5)))


    # table = f"""
    #     | Metric   |    f1     | Precision | Recall    |   AUROC   |
    #     |----------|-----------|-----------|-----------|-----------|
    #     |          |   {f1}    |{precision}| {recall}  |   {auroc} |
    # """
    # table = '\n'.join(l.strip() for l in table.splitlines())
    # writer.add_text("table", table, 0)
    # writer.flush()
    # writer.close()
    return f1, recall, precision, auroc

def Aug(train_data, prop_keep, configs, save_model = False, save_dir = './data/diffusion_outputs10/'):
  n_epoch = configs['n_epoch']
  batch_size = configs['batch_size']
  n_T = configs['n_T']
  n_classes = configs['n_classes']
  n_feat = configs['n_feat']
  lrate = configs['lrate']
  w = configs['w']

  n= train_data.data.shape[0]
  n_gen = math.ceil((1 - prop_keep) * n)
  print(n, n_gen)

  print("training generator")
  ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
  ddpm.to(device)

  dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

  optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

  for ep in range(n_epoch):
      print(f'epoch {ep}')
      ddpm.train()

      # linear lrate decay
      optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

      pbar = tqdm(dataloader)
      loss_ema = None
      for x, c in pbar:
          optim.zero_grad()
          x = x.to(device)
          c = c.to(device)
          loss = ddpm(x, c)
          loss.backward()
          if loss_ema is None:
              loss_ema = loss.item()
          else:
              loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
          pbar.set_description(f"loss: {loss_ema:.4f}")
          optim.step()

  torch.save(ddpm.state_dict(), f"model_{ep}.pth")
  torch.cuda.empty_cache()


  print("augmentation")
  #ddpm = ddpm.to("cpu") #remove for zuber
  ddpm.eval()
  with torch.no_grad():
      x_gen, x_gen_store = ddpm.sample(n_gen, (1, 28, 28), "cuda:0", label=[0],guide_w=0.5) #set to "cuda:0" for zuber
  plt.imshow(x_gen[0].reshape(28,28).cpu(), cmap="gray")
  plt.show()

  train_data.data = torch.cat([x_gen, train_data.data], 0)
  train_data.targets = torch.cat([torch.ones(n_gen),train_data.targets], 0)

  return train_data

def Aug_SMOTE(train):
    """
    require torch.dataset object
    """
    dta = torchvision.datasets.MNIST(download = FALSE)
    smote = SMOTE()
    X, y = smote.fit_resample(train.data.view(len(train), -1), train.targets) # smote the dataset (must flatten to 2d first)

    X = np.reshape(X, (len(X), 28, 28)) # reshape X to 3d

    X_tensor = torch.from_numpy(X).view(len(X), 28, 28).float().requires_grad_(True) #.to(device) # push X to GPU and reshape
    y_tensor = torch.from_numpy(y).type(torch.LongTensor) #.to(device)
    dta.data = X_tensor
    dta.targets = y_tensor

    return dta

train, test = imbalance_data(train,test,1)
train_classifier(train,test,configs)


"""
dta = torchvision.datasets.MNIST(download = FALSE)
bal_dta = torchvision.datasets.MNIST(download = FALSE) #make bal_data a torch dataset
for trial in range(1):
    dta.data, dta.targets = imbalance_data(train,test, .1) #treatment1

    n_samples = len(imb_data) #treatment0
    bal_dta.data = train.data[0:n_samples]
    bal_dta.targets = train.targets[0:n_samples]

    aug_data = Aug(dta, .1, configs_DDPM) #treatment2

    SMOTE_data = Aug_SMOTE(dta)

    train_classifier(imb_data,test,configs)
    train_classifier(bal_data,test,configs)
    train_classifier(aug_data,test,configs)
    train_classifier(SMOTE_data,test,configs)
"""





#os.system(tensorboard --logdir==runs)
end_time = time.time()
print("Time Elapsed: ", end_time - start_time)