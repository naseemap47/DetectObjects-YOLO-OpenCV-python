import cv2
from utils import findObjects

##############
# Parameters
w_h_target = 320
confThreshold = 0.6
nmsThreshold = 0.3  # value reduces, accuracy increases
###############

cap = cv2.VideoCapture(0)

classNames = []
with open('coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').rsplit('\n')
# print(classNames)

modelCfg = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNet(modelCfg, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255, (w_h_target, w_h_target),
        [0, 0, 0], 1, crop=False
    )
    net.setInput(blob)
    layerNames = net.getLayerNames()
    # print(layerNames)

    # Unconnected Out Layers (prediction Layers)
    outLayerIndex = net.getUnconnectedOutLayers()
    # print(outLayerIndex)
    outLayers = []
    for i in outLayerIndex:
        outLayers.append(layerNames[i - 1])
    # print(outLayers)

    outputs = net.forward(outLayers)
    bbox, classId, conf = findObjects(outputs, img, confThreshold=confThreshold)
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    cv2.rectangle(
        img, (x, y), (x + w, y + h),
        (255, 0, 255), 3
    )
    cv2.putText(
        img, f'{classNames[classId]} {int(conf*100)}%', (x, y-6),
        cv2.FONT_HERSHEY_PLAIN, 1.5,
        (0, 255, 0), 2
    )
    
    cv2.imshow("Web-cam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
