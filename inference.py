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

    def inference(self):

        while True:
            success, frame = self.video_capture.read()

            crop_image = frame[self.bbox[1]: self.bbox[1] + self.bbox[-1],
                         self.bbox[0]: self.bbox[0] + self.bbox[2]]
            detecting = cv2.rectangle(frame, (self.bbox[0], self.bbox[1]),
                                      (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[-1]), (0, 225, 0), 1)

            result = [str(datetime.now()), self.rec(crop_image)]
            self.list_result.append(result)
            cv2.imshow("frame", detecting)

            if cv2.waitKey(20) == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()


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
