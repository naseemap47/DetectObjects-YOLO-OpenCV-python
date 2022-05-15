import cv2

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
print(net)

while True:
    success, img = cap.read()

    cv2.imshow("Web-cam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
