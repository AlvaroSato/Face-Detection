# Bibliotecas
from ultralytics import YOLO
from torchvision import transforms
import torch
from pycocotools.coco import COCO
import cv2
import os 
import numpy as np


# Arquivos .py
import yolo
import ssd
import faster_rcnn as frcnn

def initialize():
    # Inicializando o modelo YOLO
    yolo_model = YOLO("yolov8n-face.pt")

    # Inicializando o modelo SSD
    prototxt_path = "./face_deploy.prototxt"
    model_path = "./res10_300x300_ssd_iter_140000.caffemodel"
    ssd_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Inicializando o modelo Faster R-CNN
    frcnn_model = torch.load('model_face_bounding.pth')
    device = torch.device("cuda") # Utiliza a GPU para o treinamento
    frcnn_model = frcnn_model.to(device)
    # Setando o modelo para o modo de inferência
    frcnn_model.eval()
    torch.cuda.empty_cache()

    return yolo_model, ssd_model, frcnn_model, device

def bbox_draw(img, left_coor, right_coor, conf, color, thickness, label):
    cv2.rectangle(img, (left_coor[0], left_coor[1]), (right_coor[0], right_coor[1]), color, thickness)
    if(label):
        label = label % conf
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(img, (left_coor[0], left_coor[1] - labelSize[1]),
                        (left_coor[0] + labelSize[0], left_coor[1] + baseLine),
                        color, cv2.FILLED)

        cv2.putText(img, label, left_coor,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    
    return img
    
if __name__ == "__main__":
    yolo_model, ssd_model, frcnn_model, device = initialize()

    dataset = "./dataset/detection/"
    dataset_path = os.listdir(dataset)
    dataset_len = len(dataset_path)

    print(dataset_len)

    for each in dataset_path:

        img = cv2.imread(dataset + each)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        convert_tensor = transforms.ToTensor()

        tensor_img = convert_tensor(img)

        bbox_yolo, scores_yolo = yolo.yolo_inference(img, yolo_model)
        bbox_ssd, scores_ssd = ssd.ssd_inference(img, ssd_model)
        bbox_frcnn, scores_frcnn = frcnn.frcnn_inference(tensor_img, frcnn_model, device)

        scores_yolo = scores_yolo.cpu().detach().numpy()

        bbox_frcnn = bbox_frcnn.cpu().detach().numpy()
        scores_frcnn = scores_frcnn.cpu().detach().numpy()

        # Detecções e Média das detecções
        bbox_left_x = 0
        bbox_left_y = 0
        bbox_right_x = 0
        bbox_right_y = 0
        conf = 0

        if(len(bbox_yolo)): # Verificação se o yolo detectou alguma face na imagem
            bbox_left_x += bbox_yolo[0][0]
            bbox_left_y += bbox_yolo[0][1]
            bbox_right_x += bbox_yolo[0][2]
            bbox_right_y += bbox_yolo[0][3]

            conf += scores_yolo[0]

            img = bbox_draw(img, [int(bbox_yolo[0][0]), int(bbox_yolo[0][1])], [int(bbox_yolo[0][2]), int(bbox_yolo[0][3])], scores_yolo[0], (255, 0, 0), 1, False) # Yolo BBox
        

        if(len(bbox_ssd)): # Verificação se o ssd detectou alguma face na imagem
            bbox_left_x += bbox_ssd[0][0]
            bbox_left_y += bbox_ssd[0][1]
            bbox_right_x += bbox_ssd[0][2]
            bbox_right_y += bbox_ssd[0][3]

            conf += scores_ssd[0]

            img = bbox_draw(img, [int(bbox_ssd[0][0]), int(bbox_ssd[0][1])], [int(bbox_ssd[0][2]), int(bbox_ssd[0][3])], scores_ssd[0], (0, 255, 0), 1, False) # SSD BBox
        

        if(len(bbox_frcnn)): # Verificação se o yolo detectou alguma face na imagem
            bbox_left_x += bbox_frcnn[0][0]
            bbox_left_y += bbox_frcnn[0][1]
            bbox_right_x += bbox_frcnn[0][2]
            bbox_right_y += bbox_frcnn[0][3]

            conf += scores_frcnn[0]

            img = bbox_draw(img, [int(bbox_frcnn[0][0]), int(bbox_frcnn[0][1])], [int(bbox_frcnn[0][2]), int(bbox_frcnn[0][3])], scores_frcnn[0], (0, 0, 255), 1, False) # FRCNN BBox


        
        # print("\n\n")
        # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
        # print(f"YOLO: {bbox_yolo}")
        # print(f"SSD: {bbox_ssd}")
        # print(f"FRCNN: {bbox_frcnn}")
        # print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n")

        bbox_left_x /= 3
        bbox_left_y /= 3

        bbox_right_x /= 3
        bbox_right_y /= 3

        conf /= 3


        img = bbox_draw(img, [int(bbox_left_x), int(bbox_left_y)], [int(bbox_right_x), int(bbox_right_y)], conf, (255, 255, 255), 2, "Face: %.4f") # Final BBox
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        savePath = os.path.join("./dataset/inference/", each)
        cv2.imwrite(savePath, img) 