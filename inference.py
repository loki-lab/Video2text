import cv2
import pandas as pd
from datetime import datetime


class Video2Text:
    def __init__(self, video_capture, bbox, rec_model):
        self.video_capture = video_capture
        self.bbox = bbox
        self.list_image_frame = []
        self.rec_model = rec_model
        self.timestamp_list = []
        self.text_list = []
        self.score_list = []
        self.list_result = []

    def timestamp(self):
        video_length_ms = self.video_capture.get(cv2.CAP_PROP_POS_MSEC)
        # fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        # frame_count = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        total_seconds = video_length_ms / 1000
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return "Time Stamp:" + str(hours).zfill(2) + ":" + str(minutes).zfill(2) + ":" + str(seconds).zfill(2)

    def inference(self):
        try:
            while True:
                success, frame = self.video_capture.read()

                crop_image = frame[self.bbox[1]: self.bbox[1] + self.bbox[-1],
                             self.bbox[0]: self.bbox[0] + self.bbox[2]]

                cv2.imshow("frame", crop_image)

                gray_crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

                bin_crop_image = cv2.adaptiveThreshold(gray_crop_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                       cv2.THRESH_BINARY, 391, 4)
                cv2.imshow("bin_crop_image", bin_crop_image)

                result = [self.timestamp(), self.rec(bin_crop_image)]
                print(result)
                self.list_result.append(result)

                if cv2.waitKey(20) == ord('p'):
                    cv2.waitKey()

                if cv2.waitKey(20) == ord('q'):
                    break

            self.video_capture.release()
            cv2.destroyAllWindows()
        except:
            print("End frame")

    def rec(self, image):
        result = self.rec_model.ocr(img=image, rec=True, det=False, cls=False)
        return result

    def write_result(self, filename):
        print(self.list_result)
        for i in self.list_result:
            self.timestamp_list.append(i[0])
            self.text_list.append(i[1][0][0][0])
            self.score_list.append(i[1][0][0][1])
        data = {"timestamp": self.timestamp_list, "text": self.text_list, "score": self.score_list}
        df = pd.DataFrame(data=data)
        df.to_csv(filename)
