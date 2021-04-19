print('Setting up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
import sudokuSolver

pathImage = cv2.VideoCapture(0)
address = "http://192.168.1.100:8080/video"
pathImage.open(address)
heightImg = 480
widthImg = 640

pathImage.set(3, widthImg)
pathImage.set(4, heightImg)
model = initializePredictionModel() ##LOAD THE CNN MODEL

isFound = 0

while True:
    success, img = pathImage.read()

    # 1. prepare the image
    img = cv2.resize(img, (widthImg, heightImg))  # Resize the image to make it a square image
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgThreshold = preProcess(img)

    # 2. Find all the contours
    imgContours = img.copy()  # copy image for display purposes (All the contours)
    imgBigContour = img.copy()  # this will contain the biggest contour
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # abobe we use external method since we need the outer contour
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)  # Draw all the detected contours
    cv2.imshow("Contours", imgContours)


    # 3. Find the biggest contour and use it as sudoku
    biggest, maxArea = biggestContour(contours)
    print(biggest)

    imgWarpColored = imgBlank
    if biggest.size != 0:
        isFound = 1


    else:
        print("No Sudoku Found")
    # cv2.imshow("Video", img)



    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    if isFound == 1:
        biggest = reorder(biggest)
        print(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)  # Draw the biggest contour
        cv2.imshow("Biggest Contour", imgBigContour)

        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgDetectedDigits = imgBlank.copy()
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("Warp Colored", imgWarpColored)

        # 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)
        print(len(boxes))
        # cv2.imshow("Sample", boxes[0])
        numbers = getPrediction(boxes, model)
        print(numbers)
        imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        # cv2.imshow("Detected Digits", imgDetectedDigits)
        # cv2.waitKey(0)

        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)  # if number>0, put 0 otherwise put 1
        # places where 1 is inserted, need to be filled
        print(posArray)

        #### 5. FIND SOLUTION OF THE BOARD
        board = np.array_split(numbers, 9)
        print(board)
        try:
            sudokuSolver.solve(board)
        except:
            pass
        print(board)
        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
        solvedNumbers = flatList * posArray
        imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

        # Overlay the solution
        pts2 = np.float32(biggest)
        pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgInvwarpColored = img.copy()
        imgInvwarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
        inv_perspective = cv2.addWeighted(imgInvwarpColored, 1, img, 0.5, 1)
        imgDetectedDigits = drawGrid(imgDetectedDigits)
        imgSolvedDigits = drawGrid(imgSolvedDigits)



        imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                      [imgDetectedDigits, imgSolvedDigits, imgWarpColored, inv_perspective])

        stackedImage = stackImages(0.7, imageArray)
        cv2.imshow("stacked images", stackedImage)

    else:
        print("No Sudoku Found")

    cv2.waitKey(0)


















