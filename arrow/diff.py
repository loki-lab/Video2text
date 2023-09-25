import cv2
import numpy as np

pattern_image01 = cv2.imread("../../up_down/down.jpg")
pattern_image02 = cv2.imread("../../up_down/up.jpg")
pattern_image01 = cv2.cvtColor(pattern_image01, cv2.COLOR_BGR2GRAY)
pattern_image02 = cv2.cvtColor(pattern_image02, cv2.COLOR_BGR2GRAY)

bbox = [790, 350, 100, 100]


# bbox = [530, 420, 100, 100]


def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse_result = err / (float(h * w))
    return mse_result


def read_video(path_video, bbox):
    video = cv2.VideoCapture(path_video)
    while True:
        _, frame = video.read()

        crop_image = frame[bbox[1]: bbox[1] + bbox[-1],
                     bbox[0]: bbox[0] + bbox[2]]

        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        detecting = cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[-1]), (0, 225, 0), 1)

        error01 = mse(img1=pattern_image01, img2=crop_image)

        error02 = mse(img1=pattern_image02, img2=crop_image)
        print("image video - pattern image 01 (down):", error01)
        print("image video - pattern image 02 (up):", error02)

        if error01 < error02:
            print("down")
        else:
            print("up")

        cv2.imshow("new_frame", crop_image)
        cv2.imshow("detec", detecting)
        cv2.waitKey()
        if cv2.waitKey(20) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# read_video("data/Smaple_Doppler.mp4", [790, 350, 100, 100])

read_video("data/Sample_Doppler_2.mp4", [530, 420, 100, 100])
