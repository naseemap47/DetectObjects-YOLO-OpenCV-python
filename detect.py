import cv2
from utils import findObjects

##############
# Parameters
w_h_target = 320
confThreshold = 0.6
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
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    bbox, classId, conf = findObjects(outputs, img, confThreshold=confThreshold)
    # More than one boxes coming from Output
    for i in bbox:
        x, y, w, h = i[0], i[1], i[2], i[3]
        cv2.rectangle(
            img, (x, y), (x + w, y + h),
            (0, 255, 0), 1
        )

    # print(bbox, classId, conf)

    cv2.imshow("Web-cam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
