import cv2
import pandas as pd
import re
import numpy as np

pattern_image01 = cv2.imread("arrow/up_down/down.jpg")
pattern_image02 = cv2.imread("arrow/up_down/up.jpg")
pattern_image01 = cv2.cvtColor(pattern_image01, cv2.COLOR_BGR2GRAY)
pattern_image02 = cv2.cvtColor(pattern_image02, cv2.COLOR_BGR2GRAY)


def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse_result = err / (float(h * h))
    return mse_result


class Video2Text:
    def __init__(self, video_capture, bbox, arrow_bbox, rec_model):
        self.video_capture = video_capture
        self.bbox = bbox
        self.arrow_bbox = arrow_bbox
        self.list_image_frame = []
        self.rec_model = rec_model
        self.timestamp_list = []
        self.text_list = []
        self.score_list = []
        self.list_result = []
        self.arrow_direction_list = []

    def timestamp(self):
        video_length_ms = self.video_capture.get(cv2.CAP_PROP_POS_MSEC)
        # fps = self.video_capture.get(cwv2.CAP_PROP_FPS)
        # frame_count = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        total_seconds = video_length_ms / 1000
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return "Time Stamp:" + str(hours).zfill(2) + ":" + str(minutes).zfill(2) + ":" + str(seconds).zfill(2)

    def inference(self):
        i = 0
        try:
            while True:
                i += 1
                success, frame = self.video_capture.read()
                # dread two number
                crop_image = frame[self.bbox[1]: self.bbox[1] + self.bbox[-1],
                             self.bbox[0]: self.bbox[0] + self.bbox[2]]

                cv2.imshow("crop image", crop_image)

                gray_crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

                bin_crop_image = cv2.adaptiveThreshold(gray_crop_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                       cv2.THRESH_BINARY, 391, 4)
                cv2.imshow("bin_crop_image", bin_crop_image)

                detecting = cv2.rectangle(frame, (self.bbox[0], self.bbox[1]),
                                          (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[-1]), (0, 225, 0), 1)

                cv2.imshow("frame", detecting)

                rec_value = self.rec(bin_crop_image)
                result = [self.timestamp(), rec_value]
                print(result)
                self.list_result.append(result)

                # arrow
                crop_image_arrow = frame[self.arrow_bbox[1]: self.arrow_bbox[1] + self.arrow_bbox[-1],
                                   self.arrow_bbox[0]: self.arrow_bbox[0] + self.arrow_bbox[2]]
                crop_image_arrow = cv2.cvtColor(crop_image_arrow, cv2.COLOR_BGR2GRAY)
                detecting_arrow = cv2.rectangle(frame, (self.arrow_bbox[0], self.arrow_bbox[1]),
                                                (self.arrow_bbox[0] + self.arrow_bbox[2],
                                                 self.arrow_bbox[1] + self.arrow_bbox[-1]), (0, 225, 0), 1)
                cv2.imshow("frame", detecting_arrow)
                error01 = mse(pattern_image01, crop_image_arrow)
                error02 = mse(pattern_image02, crop_image_arrow)

                if error01 < error02:
                    arr_result = "down"
                    print(arr_result)
                    self.arrow_direction_list.append(arr_result)
                else:
                    arr_result = "up"
                    print(arr_result)
                    self.arrow_direction_list.append(arr_result)

                if cv2.waitKey(20) == ord('p'):
                    cv2.waitKey()

                if cv2.waitKey(20) == ord('q'):
                    break

            self.video_capture.release()
            cv2.destroyAllWindows()
        except:
            print("end frame")

    def rec(self, image):
        result = self.rec_model.ocr(img=image, rec=True, det=False, cls=False)
        return result

    def write_result(self, filename):
        # print(self.list_result)
        for i in self.list_result:
            self.timestamp_list.append(i[0])
            self.text_list.append(re.findall("[0-9][.][0-9]", i[1][0][0][0])[0])
            self.score_list.append(i[1][0][0][1])
        data = {"timestamp": self.timestamp_list, "text": self.text_list, "score": self.score_list,
                "arrow direction": self.arrow_direction_list}
        df = pd.DataFrame(data=data)
        df.to_csv(filename)
