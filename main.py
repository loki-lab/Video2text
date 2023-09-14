import cv2
from paddleocr import PaddleOCR
from inference import Video2Text

video_capture = cv2.VideoCapture('Sample_Doppler_2.mp4')
model = PaddleOCR(rec_model_dir='models/en_PP-OCRv4_rec_infer', lang="en")

# Nhận diện dãy số với 6 chữ số
bbox1 = (470, 700, 750, 200)

read_video = Video2Text(video_capture, bbox1, model)
read_video.inference()
read_video.write_result("./output/output1.csv")


# nhận diện dãy số với 2 chữ số
bbox2 = (800, 400, 400, 200)

read_video = Video2Text(video_capture, bbox2, model)
read_video.inference()
read_video.write_result("./output/output2.csv")



