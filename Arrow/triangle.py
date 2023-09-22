import cv2
import numpy as np
import pickle
import pandas as pd

list_speed = []


def convert_shape(image):
    return [np.reshape(image, (100 * 100 * 3))]


model = pickle.load(open("knn_model.sav", "rb"))

video = cv2.VideoCapture("data/Sample_Doppler_2.mp4")

# bbox = [790, 350, 100, 100]
bbox = [530, 420, 100, 100]
try:
    while True:
        _, frame = video.read()

        crop_image = frame[bbox[1]: bbox[1] + bbox[-1],
                     bbox[0]: bbox[0] + bbox[2]]

        cv2.rectangle(frame, (bbox[0], bbox[1]),
                      (bbox[0] + bbox[2], bbox[1] + bbox[-1]), (0, 225, 0), 1)
        #
        cv2.imshow("frame", frame)
        gray_crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        bin_crop_image = cv2.adaptiveThreshold(gray_crop_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, 401, 12)
        bin_crop_image = cv2.cvtColor(bin_crop_image, cv2.COLOR_GRAY2BGR)
        # bin_crop_image = cv2.Canny(bin_crop_image, 100, 1000)

        # contours, hierarchy = cv2.findContours(bin_crop_image,
        #                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #
        # # cv2.imshow("frame", frame)
        # print(contours)
        # img_draw = cv2.polylines(crop_image, contours, True, (0, 255, 0), 1)
        # result = model.predict(convert_shape(image=bin_crop_image))
        result = model.predict(convert_shape(bin_crop_image))
        # print(result)
        if result[0] == 0:
            list_speed.append("down")
            print("down")
        else:
            list_speed.append("up")
            print("up")
        cv2.imshow("contours", bin_crop_image)

        if cv2.waitKey(20) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
except:
    print("end frame")

d = {"speed": list_speed}
df = pd.DataFrame(d)
df.to_csv("up_down.csv")
