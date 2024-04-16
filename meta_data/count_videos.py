import cv2
import os
from os import walk
import datetime

dict_path = "../data/Matignon-LSFv1/video/"
L_vids = next(walk(dict_path), (None, None, []))[2]
# print(L_vids)
L_duration = []
video_nb = 0
for v_id in L_vids:
    if "mp4" in v_id:
        video_path = os.path.join(dict_path, v_id)
        cap = cv2.VideoCapture(video_path)
        try:
            frames_nb = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            seconds = round(frames_nb / fps)  # calculate duration of the video
            L_duration.append(seconds)
            video_nb += 1
        except:
            print("pb with :", v_id)

Total_duration = sum(L_duration)
video_time = datetime.timedelta(seconds=Total_duration)
print(f"video time: {video_time}")
print(f"mean duration : {datetime.timedelta(seconds=Total_duration/video_nb)}")
print("nb of videos : ", video_nb)
