def yolo_inference(img, model):
    results = model(img)
    boxes = results[0].boxes
    return_list = []
    for box in boxes:
        aux = []
        aux.append(int(box.xyxy.tolist()[0][0]))
        aux.append(int(box.xyxy.tolist()[0][1]))
        aux.append(int(box.xyxy.tolist()[0][2]))
        aux.append(int(box.xyxy.tolist()[0][3]))

        return_list.append(aux)
    return return_list, results[0].boxes.conf