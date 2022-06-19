import cv2
import numpy as np
import argparse
from sudoku_detection import pipeline
from digit_recognizer_model import DigitRecognizerModel
import cpp_library.build.sudoku_solver as ss
from sudoku_draw import draw_on_image
from constants import BIGGER_IMG_SIZE, IMG_SIZE

parser = argparse.ArgumentParser(description='video input, video output')
parser.add_argument("input_file", help="Path to input video file")
parser.add_argument("output_file", help="Path to output video file")
parser.add_argument("--fps", default=30, help="FPS of output_file", dest="fps")
args = parser.parse_args()

# print(args.input_file)
cap = cv2.VideoCapture(args.input_file)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit(-1)

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 500
# print(WINDOW_HEIGHT, WINDOW_WIDTH)
out = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), args.fps,
                      (WINDOW_WIDTH, WINDOW_HEIGHT*2))
model = DigitRecognizerModel()
prev = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
frame = 0
while cap.isOpened():
    print("Current frame:", frame)
    frame += 1
    # Use urllib to get the image from the IP camera
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))

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
        #
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
            prev = cv2.warpPerspective(solved_warped, matrix, (WINDOW_WIDTH, WINDOW_HEIGHT), flags=cv2.WARP_INVERSE_MAP)
            # cv2.imshow("AR", image_result)
            # prev = image_result
    mask = np.where(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) != 0)
    output_image = img.copy()
    output_image[mask] = prev[mask]
    cv2.putText(img, "Input", (20, 60), cv2.FONT_ITALIC, fontScale=2, color=(255, 255, 255), thickness=2)
    cv2.putText(output_image, "Output", (20, 60), cv2.FONT_ITALIC, fontScale=2, color=(255, 255, 255), thickness=2)

    # cv2.imshow("IPWebcam", img)
    output_image = np.concatenate((img, output_image), axis=0)
    out.write(output_image)
    # cv2.imshow("IPWebcam", output_image)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
