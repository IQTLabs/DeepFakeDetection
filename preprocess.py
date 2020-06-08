import torch
import sys
import yaml
import argparse
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.insert(1, './dfdet/BlazeFace/')

from dfdet import VideoReader, FaceExtractor
from blazeface import BlazeFace

parser = parser = argparse.ArgumentParser(description='Batch inference script')

parser.add_argument('--gpu', dest='gpu', default=0,
                    type=int, help='Target gpu')
parser.add_argument('--config', dest='config',
                    default='./config_files/preprocess.yaml', type=str,
                    help='Config file with paths and MTCNN set-up')
parser.add_argument('--df', dest='df', default=None,
                    type=str, help='Dataframe location')
parser.add_argument('--debug', dest='debug', default=False,
                    type=bool, help='Debug mode switch')


def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


def preprocess_video(video_path, save_path, input_size=150):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)

        if len(faces) > 0:
            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = make_square_image(resized_face)
                    im = Image.fromarray(resized_face)
                    im.save('{}/frame_{}.png'.format(save_path, n))
                    n += 1
        return n
    except:
        return 0


def preprocess_on_video_set(df, num_workers, input_size=150):
    data_path = '/home/mlomnitz/Documents/DFDC/DeepFakeDetection/deepfake-detection/dfdc_train_all'
    save_path = '/home/mlomnitz/Documents/DFDC/DeepFakeDetection/deepfake-detection/dfdc_preprocess_blaze'
    input_size = input_size

    def process_file(i):
        entry = df.iloc[i]

        filename = '{}/{}/{}'.format(data_path, entry['split'], entry['File'])
        outdir = '{}/{}/{}'.format(save_path, entry['split'], entry['File'])
        Path(outdir).mkdir(parents=True, exist_ok=True)

        n_frames = preprocess_video(filename, outdir, input_size)
        meta = {'File': outdir, 'label': entry['label'], 'n_frames': n_frames}
        return meta

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        meta = tqdm(ex.map(process_file, range(len(df))), total=len(df))

    return pd.DataFrame(meta)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.df is not None, 'Need to specify metadata file'
    with open(args.config) as f:
        config = yaml.load(f)
    device = torch.device('cuda:{}'.format(args.gpu))

    df = pd.read_csv(args.df)
    path = config['data_path']
    # Facedet
    facedet = BlazeFace().to(device)
    facedet.load_weights("./dfdet/BlazeFace/blazeface.pth")
    facedet.load_anchors("./dfdet/BlazeFace/anchors.npy")
    _ = facedet.train(False)
    #
    video_reader = VideoReader()
    def video_read_fn(x): return video_reader.read_frames(
        x, num_frames=config['n_frames'])
    face_extractor = FaceExtractor(video_read_fn, facedet)
    faces_dataframe = preprocess_on_video_set(df, 4)
    faces_dataframe.to_csv('{}/faces_metadata.csv'.format(config['out_path']))
