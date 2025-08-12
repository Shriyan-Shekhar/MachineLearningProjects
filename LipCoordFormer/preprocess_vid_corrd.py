import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.nn import Module
import numpy as np
import tempfile
import face_alignment
from collections import OrderedDict

from utils import *
from getcoords import get_coord_pipeline
#from data import load_video

import glob
import json
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_video(video_tensor, coords_tensor, output_path):
    """Save processed video tensor as compressed numpy file"""
    np.savez_compressed(output_path, video=video_tensor.numpy(), coords=coords_tensor.numpy())

def load_video(file):
    p = tempfile.mkdtemp()
    cmd = 'ffmpeg -i \'{}\' -qscale:v 2 -r 25 \'{}/%d.jpg\''.format(file, p)
    os.system(cmd)
    
    files = os.listdir(p)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
        
    array = [cv2.imread(os.path.join(p, file)) for file in files]
    
    
    array = list(filter(lambda im: not im is None, array))
    #array = [cv2.resize(im, (100, 50), interpolation=cv2.INTER_LANCZOS4) for im in array]
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
    points = [fa.get_landmarks(I) for I in array]
    
    front256 = get_position(256)
    video = []
    for point, scene in zip(points, array):
        if(point is not None):
            shape = np.array(point[0])
            shape = shape[17:]
            M = transformation_from_points(np.matrix(shape), np.matrix(front256))
           
            img = cv2.warpAffine(scene, M[:2], (256, 256))
            (x, y) = front256[-20:].mean(0).astype(np.int32)
            w = 160//2
            img = img[y-w//2:y+w//2,x-w:x+w,...]
            img = cv2.resize(img, (128, 64))
            video.append(img)
    
    #Save the whole video
    
    
    
    
    
    vid_len = len(video)
    video = np.stack(video, axis=0).astype(np.float32)
    
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0
    
    
    
    
    #video = video.unsqueeze(0)
    return video, p, vid_len


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def preprocess(speaker, max_videos=1):
    #speaker = ["s1", ...]
    # Get all video files in the directory
    video_files = glob.glob(f"../data/grid/{speaker}/*.mpg")
    #output_path = f"../data/grid/{speaker}.json"
    
    #if not os.path.exists(output_path):
    #    with open(output_path, 'w') as f:
    #        json.dump({}, f)


    print(f"Found {len(video_files)} video files for speaker {speaker}.")

    # Create a directory sX_processed if not exist
    if not os.path.exists(f"../data/grid/{speaker}_processed"):
        os.makedirs(f"../data/grid/{speaker}_processed")
    
    # Process videos incrementally
    for i, video_path in enumerate(video_files):
        
        print(f"Processing {i+1}/{len(video_files)}: {video_path}")
        
        # Process new video
        video, _, _ = load_video(video_path)
        
        #get video name from video_path
        video_name = os.path.splitext(os.path.basename(video_path))[0] 

        coords = get_coord_pipeline(video_path, device)[1]
        
        
        save_video(video, coords, f"../data/grid/{speaker}_processed/{video_name}.npz")
        
        '''
        vid_data = {  
            "coords": coords.tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(vid_data, f, cls=NumpyEncoder, indent=2)
        '''
    
    print(f"Processing complete. Results.")
    
  
def load_processed_vid(video_path):
    loaded = np.load(f"{video_path}.npz")
    video_tensor = torch.from_numpy(loaded['video'])    
    coords_tensor = torch.from_numpy(loaded['coords'])
    
    #print(video_tensor)
    #print(f"Video shape: {video_tensor.shape}")
    #print(f"Coords shape: {coords_tensor.shape}")
    
    return video_tensor, coords_tensor
    
if __name__ == "__main__":
    # Example usage
    speakers = ["s6", "s7", "s8", "s9", "s10"]
    
    for speaker in speakers:
    
        print(f"Now going to preprocess for speaker {speaker}")
        
        start = time.time()
        
        preprocess(speaker)
        
        end = time.time()
        
        print(f"Time used: {end - start}")
        
    #load_processed_vid("../data/grid/s5_processed/bbaf2a")
    #video, _, _ = load_video("../data/grid/s1/sgap2p.jpg")
    
    #print(video)