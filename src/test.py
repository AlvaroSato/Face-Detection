# Data processing
import numpy as np
import pandas as pd

import os
import sys

# Pytorch
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from torchvision.utils import draw_bounding_boxes, save_image

# Image processing libs
import copy
import math
from PIL import Image
import cv2

# Adicional libs
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
import albumentations as A  

import matplotlib.pyplot as plt
# %matplotlib inline

# COCO tools
# Para visualização do arquivo "annotations.coco"
from pycocotools.coco import COCO
# Transformações utilizando albumentations
from albumentations.pytorch import ToTensorV2

def get_transforms():
    transform = A.Compose([
            A.Resize(600, 600), # Redimensionar a imagem 
            ToTensorV2() # Conversão da imagem para o tipo "torch.tensor"
        ], bbox_params=A.BboxParams(format='coco'))
    return transform

# Classe para a criação do dataset (Treinamento e para realizar a detecção da face)
class FaceDetection(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, targetTransform=None, transforms=get_transforms()):
        super().__init__(root, transforms, transform, targetTransform)
        self.split = split # Split do diretório do dataset - EX: Dataset/train por padrão
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json")) # Arquivo annotations.coco do dataset
        self.ids = list(sorted(self.coco.imgs.keys())) # IDS, no arquivo "annotations.coco" cada imagem recebe um ID
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
    
    # Realiza a abertura da imagem
    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]['file_name'] # Nome do arquivo
        image = cv2.imread(os.path.join(self.root, self.split, path)) # Realiza a abertura da imagem com o opencv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converte a imagem de BGR para RGB
        return image
    
    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    # Função para realizar a extração de todas as variáveis necessárias do annotations.coco
    def __getitem__(self, index):
        id = self.ids[index] # ID da imagem
        image = self._load_image(id) # Carrega a imagem
        target = self._load_target(id) # Imagem alvo
        target = copy.deepcopy(self._load_target(id)) # Copia da imagem
        
        boxes = [t['bbox'] + [t['category_id']] for t in target] # Bounding box das faces da imagem

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
        
        image = transformed['image']
        boxes = transformed['bboxes']
        
        # Realiza a conversão do width e height para coordenadas x e y do canto oposto ao ponto inicial do bounding box
        new_boxes = [] # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        # Realiza a conversão para tensor, para o pytorch
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        # Dict para todas as variáveis obtidas
        targ = {} 
        targ['boxes'] = boxes # Bounding Boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64) # Label do bounding box
        targ['image_id'] = torch.tensor([t['image_id'] for t in target]) # Id da imagem alvo
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # Área do bounding box
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64) # Mais de uma face na imagem
        return image.div(255), targ # Realiza a reescala da imagem
    
    # Função para saber a quantidade de imagens no dataset de treinamento ou para detecção
    def __len__(self):
        return len(self.ids)
    

train_dataset = FaceDetection(root="dataset/Faster_RCNN/")
# Limpando a memória antes da criação do modelo
import gc
gc.collect()
torch.cuda.empty_cache()

# Carregando o modelo do nosso faster Rcnn
model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
# Realizando a troca das layers de saída
# Já que o modelo MobileNetV3-Large foi treinado com 90 classes, e o nosso modelo precisa detectar apenas uma classe
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 1)

# Função para o agrupamento para o carregamento dos dados de treinamento
# Permitindo criarmos batches para o treinamento do modelo
def collate_fn(batch):
    return tuple(zip(*batch))

# Realizando o carregamento das imagens para o treinamento
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# As linhas de código presente nesta célula
# Garante que o modelo irá realizar o treinamento e não irá crashar durante o treinamento

images,targets = next(iter(train_loader))
images = list(image for image in images)
targets = [{k:v for k, v in t.items()} for t in targets]
output = model(images, targets)
device = torch.device("cuda") # Utiliza a GPU para o treinamento
model = model.to(device)


# Optimizador para o treinamento
# Utilizando Stochastic Gradient Descent
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1) # lr scheduler

# Função para realizar o treinamento de um epoch
# Número de epochs definidos pelo usuário
def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train() # Treina o modelo
    
    # Loss de cada epoch do treinamento
    all_losses = []
    all_losses_dict = []

    for images, targets in tqdm(loader): # Progressive bar (tqdm)
        # Imagens para o treinamento
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
        # Cálculo da loss 
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        # Armazenando a loss
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        # Realiza a verificação se a loss se tornou infinita, e para o treinamento
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig")
            print(loss_dict)
            sys.exit(1)
        
        # Optimizador do treinamento
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    # Informações demonstradas para o usuário a cada epoch do treinamento
    all_losses_dict = pd.DataFrame(all_losses_dict)
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))
    
    return np.mean(all_losses)


epoch = 0
loss = 1

# Realiza o treinamento do modelo baseado no numero de epochs escolhido pelo usuário
start_time = time.time()
while(loss > 0.15):
    loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
    epoch = epoch + 1
#     lr_scheduler.step()
print("CNN - %s seconds" % (time.time() - start_time))

torch.save(model.state_dict(), './model.pth')