import urllib.request
import cv2
import numpy as np


from sudoku_detection import pipeline
from digit_recognizer_model import DigitRecognizerModel
import cpp_library.build.sudoku_solver as ss
from sudoku_draw import draw_on_image
from constants import BIGGER_IMG_SIZE, IMG_SIZE

WINDOW_HEIGHT = 592
WINDOW_WIDTH = 1000

model = DigitRecognizerModel()
# Replace the URL with your own IPwebcam shot.jpg IP:port
url = "http://192.168.0.100:8080/shot.jpg"




prev = np.zeros((WINDOW_HEIGHT,WINDOW_WIDTH,3), np.uint8)

while True:
    # Use urllib to get the image from the IP camera
    imgResponse = urllib.request.urlopen(url)
    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)

    # Decode the array to OpenCV usable format
    img = cv2.imdecode(imgNp, -1)
    img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    # put the image on screen

    detected = pipeline(img)
    if detected is not None:
        img_for_contour = img.copy()
        contour = np.array(detected).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(img_for_contour, [contour], -1, (0, 0, 255), 3)
        # cv2.imshow("Biggest Contour", img_for_contour)

        pts1 = np.array(detected).astype("float32")
        pts2 = np.float32([[0, 0], [BIGGER_IMG_SIZE, 0], [0, BIGGER_IMG_SIZE], [BIGGER_IMG_SIZE, BIGGER_IMG_SIZE]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, matrix, (BIGGER_IMG_SIZE, BIGGER_IMG_SIZE))
        # cv2.imshow("Warped", warped)
        recognized = model.predict(warped, 3)

        sudoku = ss.SudokuDeck(3)
        for row in range(9):
            for col in range(9):
                sudoku.set(row, col, int(recognized[row * 9 + col]))
        ok = sudoku.solve()
        if ok:
            solved = []
            for row in range(9):
                for col in range(9):
                    solved.append(sudoku.get(row, col))
            solved = np.array(solved)
            solved_warped = draw_on_image(warped, solved - recognized, 3)
            # cv2.imshow("Solved", solved_warped)
            # image_result = cv2.warpPerspective(solved_warped, matrix, (1000, 592),dst=img, flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT)
            prev = cv2.warpPerspective(solved_warped, matrix, (1000, 592),flags=cv2.WARP_INVERSE_MAP)
            # cv2.imshow("AR", image_result)
            # prev = image_result
    mask = np.where(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) != 0)
    img[mask] = prev[mask]
    cv2.imshow("IPWebcam", img)

    # Program closes if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
