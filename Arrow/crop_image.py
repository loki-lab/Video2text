import cv2

video = cv2.VideoCapture("data/Sample_Doppler_2.mp4")

bbox = [790, 350, 100, 100]
# bbox = [530, 420, 100, 100]
i = 0
while True:
    i += 1
    _, frame = video.read()

    crop_image = frame[bbox[1]: bbox[1] + bbox[-1],
                 bbox[0]: bbox[0] + bbox[2]]

    gray_crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    bin_crop_image = cv2.adaptiveThreshold(gray_crop_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 391, 3)
    cv2.rectangle(frame, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[-1]), (0, 225, 0), 1)

    cv2.imshow("frame", bin_crop_image)
    cv2.imshow("new_frame",frame)
    cv2.imwrite("triangle_output/up/" + str(i) + ".jpg", bin_crop_image)

    if cv2.waitKey(20) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
