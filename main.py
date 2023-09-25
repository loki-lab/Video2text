import cv2
from paddleocr import PaddleOCR
from inference import Video2Text

video_capture = cv2.VideoCapture('Sample_Doppler_2.mp4')
model = PaddleOCR(rec_model_dir='models/en_PP-OCRv4_rec_infer', lang="en")

# Nhận diện dãy số với 6 chữ số
bbox2 = (750, 400, 420, 220)
a_bbox = [520, 430, 100, 100]

read_video = Video2Text(video_capture, bbox2, a_bbox, model)
read_video.inference()
read_video.write_result("./output/output2.csv")

# nhận diện dãy số với 2 chữ số
# bbox2 = (800, 400, 400, 200)
#
# read_video = Video2Text(video_capture, bbox2, model)
# read_video.inference()
# read_video.write_result("./output/output2.csv")
#
#
