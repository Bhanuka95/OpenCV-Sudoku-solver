import cv2

heightImg = 480
widthImg = 640
cap = cv2.VideoCapture(0)
address = "http://192.168.1.100:8080/video"
cap.open(address)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break