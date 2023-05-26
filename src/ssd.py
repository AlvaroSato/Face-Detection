import cv2

def ssd_inference(img, ssd_model):
    w = img.shape[1]
    h = img.shape[0]
    inWidth = 200
    inHeight = 200

    blob = cv2.dnn.blobFromImage(img, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False)
    ssd_model.setInput(blob)
    detections = ssd_model.forward()

    return_list = []
    scores = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        aux = []
        if(confidence > 0.8):
            aux.append(int(detections[0, 0, i, 3] * w))
            aux.append(int(detections[0, 0, i, 4] * h))
            aux.append(int(detections[0, 0, i, 5] * w))
            aux.append(int(detections[0, 0, i, 6] * h))

            return_list.append(aux)
            scores.append(confidence)

    
    return return_list, scores