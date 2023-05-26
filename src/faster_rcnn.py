import torch
from torchvision import datasets, models
from pycocotools.coco import COCO
import cv2
import os 
import copy
import albumentations as A  
from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_bounding_boxes, save_image
import numpy as np
import pandas as pd


def frcnn_inference(img, model, device):
    # img_int = torch.tensor(img*255, dtype=torch.uint8)

    with torch.no_grad():
        prediction = model([img.to(device)]) # Realiza a detecção, mandando a imagem para o modelo
        pred = prediction[0]
        # labels = []
        # scores = pred['scores'].cpu().detach().numpy() # Pega todos as detecções realizada pelo modelo e realiza a conversão para um lista numpy

    return pred['boxes'][pred['scores'] > 0.8], pred['scores']

# def executeInference():
#     for i in range(30):
#         # Imagem atual
#         img, _ = test_dataset[i]
#         img_int = torch.tensor(img*255, dtype=torch.uint8)
#         img_name = test.loadImgs(i)[0]['file_name']
#         img_name_split = img_name.split('.')
#         img_name = img_name_split[0]+'.jpg'
#         # Desabilitando o calculo do gradiente para redução do consumo de memória
#         with torch.no_grad():
#             prediction = model([img.to(device)]) # Realiza a detecção, mandando a imagem para o modelo
#             pred = prediction[0]
#             labels = []
#             scores = pred['scores'].cpu().detach().numpy() # Pega todos as detecções realizada pelo modelo e realiza a conversão para um lista numpy
#     #     print(len(pred['boxes'][pred['scores'] > 0.01]))

#     #     print(scores)
#         for i in range(len(scores)):
#             # Classe (Em nosso caso temos apenas uma classe, porém em caso de mais classes é ulizado uma lista )
#             classe = "Faces"
#             score = scores[i] # Score atual
#             if(score > 0.8): # Theshold do score
#                 label = f"{classe} {score:0.4f}" # Label do bounding box
#                 labels.append(label) # Append do label
#         print(len(labels))
#     #     savePath = "./Result Test/" + img_name
#         saveImg = draw_bounding_boxes(img_int,
#         pred['boxes'][pred['scores'] > 0.8],
#         [labels[i] for i in range(len(labels))],
#         width=2,
#         colors=[(255, 255, 255)], font="times.ttf", font_size=30)


#     #     img = Image.fromarray(saveImg.cpu().detach().numpy()[1])
#     #     img.save(savePath)

#         r = saveImg.cpu().numpy()[0]
#         g = saveImg.cpu().numpy()[1]
#         b = saveImg.cpu().numpy()[2]
#         rgb = np.dstack((r,g,b))
#         save = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#         savePath = os.path.join("./dataset/inference/", img_name)
#         cv2.imwrite(savePath, save)