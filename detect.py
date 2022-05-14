import cv2

cap = cv2.VideoCapture(0)

classNames = []
with open('coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').rsplit('\n')
print(classNames)

while True:
    success, img = cap.read()

    cv2.imshow("Web-cam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
