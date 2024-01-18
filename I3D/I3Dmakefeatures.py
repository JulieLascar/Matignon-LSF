import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
from pathlib import Path
from utils_demo import *
import os

def get_video_name(dossier):
     # Liste des fichiers dans le dossier
    fichiers = [os.path.splitext(filename)[0] for filename in os.listdir(dossier)]
    return fichiers


p = argparse.ArgumentParser(description="Helper script to run demo.")
p.add_argument(
    "--i3d_checkpoint_path",
    type=Path,
    default="models/i3d/bsl5k.pth.tar",
    help="Path to I3D checkpoint.",
)

p.add_argument(
    "--video_path",
    type=Path,
    help="Path to test video.",
)

p.add_argument(
    "--feature_path",
    type=Path,
    help="Path to pre-extracted features.",
)

p.add_argument(
    "--num_in_frames",
    type=int,
    default=16,
    help="Number of frames processed at a time by the model",
)
p.add_argument(
    "--stride",
    type=int,
    default=1 ,
    help="Number of frames to stride when sliding window.",
)
p.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Maximum number of clips to put in each batch",
)
p.add_argument(
    "--fps",
    type=int,
    default=25,
    help="The frame rate at which to read the video",
)
p.add_argument(
    "--num_classes",
    type=int,
    default=5383,
    help="The number of classes predicted by the I3D model",
)

p.add_argument(
    "--save_features",
    default = True,
    action='store_true',
    help="Save I3D features of the video",
)

cpt = 0

# Exemple d'utilisation
dossier_test = "../../mediapi/data/mediapi_rgb/video_crops_test/"
video_names = get_video_name(dossier_test)
# print(video_names)

# video_names = ['ffcc270e75_0006','cr']
#video_names = ['ffcc270e75_0006']
# video_names = ["zwr3T94t0ZE_clip_cropped"]

# video_names = ['V-XRefvjCEU_clip_cropped', 'Vyl8jrHGA6k_clip_cropped', 'VZeIXQAt2C8_clip_cropped', 'wMDNQG0oA7k_clip_cropped', 'X3SSQD7qsTw_clip_cropped', 'YmXGfLxTvFQ_clip_cropped', 'Yw0atlzQbEs_clip_cropped']
for video_name in video_names:
    if video_name != "X3SSQD7qsTw_clip_cropped" and video_name !="awTzYOLwLSc_clip_cropped" and video_name !="FDNmSH37IEk_clip_cropped" and video_name!='V6VNZD5jLH8_clip_cropped':
        cpt +=1
        print(cpt)
        video_path = '../../mediapi/data/mediapi_rgb/video_crops_test/' + video_name + ".mp4"
        #video_path = '../../mediapi/data/mediapi_rgb/video_crops_train/' + video_name + ".mp4"
        args = p.parse_args(['--video_path', video_path, '--feature_path', 'features/mediapi/' + video_name + '.mat'])
        print('args :', args.video_path)
        print('args :', args.feature_path)
        features = main_i3d(**vars(args))
    else:
        print("not a video : X3SSQD7qsTw_clip_cropped")
