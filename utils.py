import cv2.dnn
import numpy as np


def findObjects(outputs, img, confThreshold=0.6, nmsThreshold=0.3):
    h_img, w_img, channel = img.shape
    bbox = []
    classIds = []
    conf = []

    for output in outputs:
        for detect in output:
            score = detect[5:]
            classId = np.argmax(score)
            confidence = score[classId]
            if confidence > confThreshold:
                w, h = int(detect[2] * w_img), int(detect[3] * h_img)
                x, y = int((detect[0] * w_img) - w_img / 2), int((detect[1] * h_img) - h_img / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                conf.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, conf, confThreshold, nmsThreshold)
    indices = indices.astype(int)
    # print(indices)
    bbox = np.array(bbox)[indices.astype(int)][0]
    # bbox = bbox[indices] # It will Throw An ERROR
    classIds = np.array(classIds)[indices.astype(int)][0]
    conf = np.array(conf)[indices.astype(int)][0]
    return bbox, classIds, conf
