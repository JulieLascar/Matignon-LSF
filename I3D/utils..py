import numpy as np
import torch
import cv2
import scipy.misc
import scipy.ndimage
import scipy.io
import os
from zsvision.zs_utils import BlockTimer
import shutil
from pathlib import Path
import math
from beartype import beartype
import models
import tqdm


def torch_to_list(torch_tensor):
    return torch_tensor.cpu().numpy().tolist()


def save_pred(preds, checkpoint="checkpoint", filename="preds_valid.mat"):
    preds = to_numpy(preds)
    checkpoint.mkdir(exist_ok=True, parents=True)
    filepath = os.path.join(checkpoint, filename)
    mdict = {"preds": preds}
    print(f"Saving to {filepath}")
    scipy.io.savemat(filepath, mdict=mdict, do_compression=False, format="4")


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    return tensor


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def resize_generic(img, oheight, owidth, interp="bilinear", is_flow=False):
    """
    Args
    inp: numpy array: RGB image (H, W, 3) | video with 3*nframes (H, W, 3*nframes)
          |  single channel image (H, W, 1) | -- not supported:  video with (nframes, 3, H, W)
    """

    # resized_image = cv2.resize(image, (100, 50))
    ht, wd, chn = img.shape[0], img.shape[1], img.shape[2]
    if chn == 1:
        resized_img = scipy.misc.imresize(img.squeeze(), [oheight, owidth], interp=interp, mode="F").reshape(
            (oheight, owidth, chn)
        )
    elif chn == 3:
        # resized_img = scipy.misc.imresize(img, [oheight, owidth], interp=interp)  # mode='F' gives an error for 3 channels
        resized_img = cv2.resize(img, (owidth, oheight))  # inverted compared to scipy
    elif chn == 2:
        # assert(is_flow)
        resized_img = np.zeros((oheight, owidth, chn), dtype=img.dtype)
        for t in range(chn):
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp)
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp, mode='F')
            # resized_img[:, :, t] = np.array(Image.fromarray(img[:, :, t]).resize([oheight, owidth]))
            resized_img[:, :, t] = scipy.ndimage.interpolation.zoom(img[:, :, t], [oheight, owidth])
    else:
        in_chn = 3
        # Workaround, would be better to pass #frames
        if chn == 16:
            in_chn = 1
        if chn == 32:
            in_chn = 2
        nframes = int(chn / in_chn)
        img = img.reshape(img.shape[0], img.shape[1], in_chn, nframes)
        resized_img = np.zeros((oheight, owidth, in_chn, nframes), dtype=img.dtype)
        for t in range(nframes):
            frame = img[:, :, :, t]  # img[:, :, t*3:t*3+3]
            frame = cv2.resize(frame, (owidth, oheight)).reshape(oheight, owidth, in_chn)
            # frame = scipy.misc.imresize(frame, [oheight, owidth], interp=interp)
            resized_img[:, :, :, t] = frame
        resized_img = resized_img.reshape(resized_img.shape[0], resized_img.shape[1], chn)

    if is_flow:
        # print(oheight / ht)
        # print(owidth / wd)
        resized_img = resized_img * oheight / ht
    return resized_img


def color_normalize(x, mean, std):
    """Normalize a tensor of images by subtracting (resp. dividing) by the mean (resp.
    std. deviation) statistics of a dataset in RGB space.
    """
    if x.dim() in {3, 4}:
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        assert x.size(0) == 3, "For single video format, expected RGB along first dim"
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
    elif x.dim() == 5:
        assert x.shape[1] == 3, "For batched video format, expected RGB along second dim"
        x[:, 0].sub_(mean[0]).div_(std[0])
        x[:, 1].sub_(mean[1]).div_(std[1])
        x[:, 2].sub_(mean[2]).div_(std[2])
    return x


def load_rgb_video_GULVAROL(video_path: Path, fps: int) -> torch.Tensor:
    """
    Load frames of a video using cv2 (fetch from provided URL if file is not found
    at given location).
    """
    cap = cv2.VideoCapture(str(video_path))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # cv2 won't be able to change frame rates for all encodings, so we use ffmpeg
    if cap_fps != fps:
        tmp_video_path = f"{video_path}.tmp.{video_path.suffix}"
        shutil.move(video_path, tmp_video_path)
        cmd = f"ffmpeg -i {tmp_video_path} -pix_fmt yuv420p " f"-filter:v fps=fps={fps} {video_path}"
        print(f"Generating new copy of video with frame rate {fps}")
        os.system(cmd)
        Path(tmp_video_path).unlink()
        cap = cv2.VideoCapture(str(video_path))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        assert cap_fps == fps, f"ffmpeg failed to produce a video at {fps}"

    f = 0
    rgb = []
    while True:
        # frame: BGR, (h, w, 3), dtype=uint8 0..255
        ret, frame = cap.read()
        if not ret:
            break
        # BGR (OpenCV) to RGB (Torch)
        frame = frame[:, :, [2, 1, 0]]
        rgb_t = im_to_torch(frame)
        rgb.append(rgb_t)
        f += 1
    cap.release()
    # (nframes, 3, cap_height, cap_width) => (3, nframes, cap_height, cap_width)
    rgb = torch.stack(rgb).permute(1, 0, 2, 3)
    print(f"Loaded video {video_path} with {f} frames [{cap_height}hx{cap_width}w] res. " f"at {cap_fps}")
    return rgb


def load_rgb_video(video_path, fps: int) -> torch.Tensor:
    """
    Load frames of a video using cv2.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    nbf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(cap_height)
    print(cap_width)
    print(nbf)

    if nbf > 180000:
        cap.release()
        return torch.zeros(1)

    else:
        # cv2 won't be able to change frame rates for all encodings, so we use ffmpeg
        if cap_fps != fps:
            tmp_video_path = f"{video_path}.tmp.{video_path.suffix}"
            print("HERE", tmp_video_path)
            shutil.move(video_path, tmp_video_path)
            cmd = f"ffmpeg -i {tmp_video_path} -pix_fmt yuv420p " f"-filter:v fps=fps={fps} {video_path}"
            print(f"Generating new copy of video with frame rate {fps}")
            os.system(cmd)
            Path(tmp_video_path).unlink()
            cap = cv2.VideoCapture(str(video_path))
            cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            assert cap_fps == fps, f"ffmpeg failed to produce a video at {fps}"

        f = 0
        rgb = []

        while True:
            # frame: BGR, (h, w, 3), dtype=uint8 0..255
            ret, frame = cap.read()
            if not ret:
                break
            # BGR (OpenCV) to RGB (Torch)
            height, width, _ = frame.shape
            # if height > 1000 or width > 1000 :
            #     new_h = int(height / 2)
            #     new_w = int(width / 2)
            #     frame = cv2.resize(frame.astype(np.uint8), (new_w, new_h), interpolation = cv2.INTER_AREA)
            if height > 224 or width > 224:
                frame = cv2.resize(frame.astype(np.uint8), (224, 224), interpolation=cv2.INTER_AREA)
            frame = frame[:, :, [2, 1, 0]]
            rgb_t = im_to_torch(frame)
            rgb.append(rgb_t)
            del frame
            f += 1

        cap.release()
        # (nframes, 3, cap_height, cap_width) => (3, nframes, cap_height, cap_width)
        rgb = torch.stack(rgb).permute(1, 0, 2, 3)
        print(f"Loaded video {video_path} with {f} frames [{cap_height}hx{cap_width}w] res. " f"at {cap_fps}")
        cap.release()
        return rgb


@beartype
def prepare_input(
    rgb: torch.Tensor,
    resize_res: int = 256,
    inp_res: int = 224,
    mean: torch.Tensor = 0.5 * torch.ones(3),
    std=1.0 * torch.ones(3),
):
    """
    Process the video:
    1) Resize to [resize_res x resize_res]
    2) Center crop with [inp_res x inp_res]
    3) Color normalize using mean/std
    """
    iC, iF, iH, iW = rgb.shape
    rgb_resized = np.zeros((iF, resize_res, resize_res, iC))
    for t in range(iF):
        tmp = rgb[:, t, :, :]
        tmp = resize_generic(im_to_numpy(tmp), resize_res, resize_res, interp="bilinear", is_flow=False)
        rgb_resized[t] = tmp
    rgb = np.transpose(rgb_resized, (3, 0, 1, 2))
    # Center crop coords
    ulx = int((resize_res - inp_res) / 2)
    uly = int((resize_res - inp_res) / 2)
    # Crop 256x256
    rgb = rgb[:, :, uly : uly + inp_res, ulx : ulx + inp_res]
    rgb = to_torch(rgb).float()
    assert rgb.max() <= 1
    rgb = color_normalize(rgb, mean, std)
    return rgb


@beartype
def load_i3d_model(
    i3d_checkpoint_path: Path,
    num_classes: int,
    num_in_frames: int,
) -> torch.nn.Module:
    """Load pre-trained I3D checkpoint, put in eval mode."""
    model = models.InceptionI3d(
        num_classes=num_classes,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=num_in_frames,
        include_embds=True,
    )
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(i3d_checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@beartype
def sliding_windows(
    rgb: torch.Tensor,
    num_in_frames: int,
    stride: int,
) -> tuple:
    """
    Return sliding windows and corresponding (middle) timestamp
    """
    C, nFrames, H, W = rgb.shape
    # If needed, pad to the minimum clip length
    if nFrames < num_in_frames:
        rgb_ = torch.zeros(C, num_in_frames, H, W)
        rgb_[:, :nFrames] = rgb
        rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
        rgb = rgb_
        nFrames = rgb.shape[1]

    num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1
    plural = ""
    if num_clips > 1:
        plural = "s"
    print(f"{num_clips} clip{plural} resulted from sliding window processing.")

    rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W)
    t_mid = []
    # For each clip
    for j in range(num_clips):
        # Check if num_clips becomes 0
        actual_clip_length = min(num_in_frames, nFrames - j * stride)
        if actual_clip_length == num_in_frames:
            t_beg = j * stride
        else:
            t_beg = nFrames - num_in_frames
        t_mid.append(t_beg + num_in_frames / 2)
        rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]
    return rgb_slided, np.array(t_mid)


def main_i3d(
    i3d_checkpoint_path: Path,
    video_path: str,
    feature_path: str,
    fps: int,
    num_classes: int,
    num_in_frames: int,
    batch_size: int,
    stride: int,
    save_features: bool,
):
    with BlockTimer("Loading I3D model"):
        model = load_i3d_model(
            i3d_checkpoint_path=i3d_checkpoint_path,
            num_classes=num_classes,
            num_in_frames=num_in_frames,
        )
    print("model is in cuda :", next(model.parameters()).is_cuda)
    # with BlockTimer("Loading video frames"):
    rgb_orig = load_rgb_video(
        video_path=video_path,
        fps=fps,
    )
    print(rgb_orig.shape)
    if len(rgb_orig) == 1:
        return torch.zeros(1)

    else:
        # Prepare: resize/crop/normalize
        rgb_input = prepare_input(rgb_orig)
        # Sliding window
        print(rgb_input.shape)
        nb_frames = rgb_input.shape[1]
        nb_fenetres = nb_frames - num_in_frames + 1

        all_features = torch.Tensor(nb_fenetres, 1024)
        for i in tqdm.tqdm(range(nb_fenetres)):
            # Forward pass
            video_window = rgb_input[:, i : i + num_in_frames, :, :]
            inp = video_window.reshape(
                1, video_window.shape[0], video_window.shape[1], video_window.shape[2], video_window.shape[3]
            )
            out = model(inp)
            all_features[i] = out["embds"].squeeze().data.cpu()
        print(all_features.shape)
        if save_features:
            scipy.io.savemat(feature_path, mdict={"preds": all_features}, do_compression=False, format="4")
        # else :
        #     # Prepare: resize/crop/normalize

        #     rgb_input = prepare_input(rgb_orig)
        #     print(rgb_input.shape)
        #     # Sliding window
        #     rgb_slides, t_mid = sliding_windows(
        #         rgb=rgb_input,
        #         stride=stride,
        #         num_in_frames=num_in_frames,
        #     )
        #     # Number of windows/clips
        #     num_clips = rgb_slides.shape[0]
        #     # Group the clips into batches
        #     num_batches = math.ceil(num_clips / batch_size)
        #     all_features = torch.Tensor(num_clips, 1024)

        #     for b in range(num_batches):
        #         inp = rgb_slides[b * batch_size : (b + 1) * batch_size]
        #         # Forward pass
        #         out = model(inp)
        #         all_features[b] = out["embds"].squeeze().data.cpu()

        #     if save_features:
        #         scipy.io.savemat(feature_path, mdict={"preds": all_features}, do_compression=False, format="4")

        return all_features
