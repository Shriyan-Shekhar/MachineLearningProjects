import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import editdistance
import json
from torch.nn import Module

import options as opt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyDataset(Dataset):
    letters = [
        " ",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    def __init__(
        self,
        video_path,
        anno_path,
        coords_path,
        file_list,
        vid_pad,
        txt_pad,
        phase,
    ):
        self.anno_path = anno_path
        self.coords_path = coords_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase

        # with open(file_list, "r") as f:
        #     self.videos = [
        #         os.path.join(video_path, line.strip()) for line in f.readlines()
        #     ]
        
        processsed_speaker = ["s4", "s5", "s6", "s7", "s8", "s9", "s10"]
        video_files = [os.path.join("../data/grid/s1_processed", f) for f in os.listdir("../data/grid/s1_processed") if f.endswith(".npz")]
        align_files = [os.path.join("../data/grid/s1_align", f) for f in os.listdir("../data/grid/s1_align") if f.endswith(".align")] 
        
        for speaker in processsed_speaker:
            video_files.extend([os.path.join(f"../data/grid/{speaker}_processed", f) for f in os.listdir(f"../data/grid/{speaker}_processed") if f.endswith(".npz")])
            align_files.extend([os.path.join(f"../data/grid/{speaker}_align", f) for f in os.listdir(f"../data/grid/{speaker}_align") if f.endswith(".align")])

        video_files.sort()
        align_files.sort()

        # print(video_files)
        # print(align_files)

        self.videos = video_files
        self.alignments = align_files          

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        align_path = self.alignments[idx]
        
        # Load video data
        loaded = np.load(video_path)
        video = torch.from_numpy(loaded['video'])
        coords = torch.from_numpy(loaded['coords']).float().unsqueeze(0)
        
        # Handle empty coords case        
        if len(coords) == 0:
            print(f"No faces detected in video: {video_path}")
            return {
                "vid": None,
                "txt": None,
                "coord": None,
                "txt_len": 0,
                "vid_len": 0,
            }

        
        # Temporal alignment
        if coords.size(1) > video.size(1):
            coords = coords[:, :video.size(1), :, :]
        elif coords.size(1) < video.size(1):
            padding = video.size(1) - coords.size(1)
            coords = torch.cat([coords, torch.zeros(1, padding, coords.size(2), coords.size(3))], dim=1)
        
        # Video processing
        video = video.permute(1, 0, 2, 3)
        video = torch.FloatTensor(self._padding(video, self.vid_pad))
        video = video.permute(1, 0, 2, 3)
        
        # Coordinates processing
        coords = coords.squeeze(0)
        coords = torch.FloatTensor(self._padding(coords, self.vid_pad))
        
        # Text processing - FIXED SECTION
        with open(align_path, "r") as f:
            lines = [line.strip().split(" ") for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ["SIL", "SP"], txt))
        
        # Use the dataset's built-in text conversion method
        joined_text = " ".join(txt).upper()
        transcript = MyDataset.txt2arr(joined_text, start=1)  # Consistent with arr2txt's start=1
        
        # Convert to tensor and pad
        txt_len = len(transcript)
        transcript = torch.LongTensor(transcript)
        transcript = self._padding(transcript, self.txt_pad)
        
        # print(f"Video shape: {video.shape}, Coords shape: {coords.shape}, Transcript shape {transcript.shape}")
        return {
            "vid": video,
            "txt": torch.LongTensor(transcript),
            "coord": coords,
            "txt_len": txt_len,
            "vid_len": video.shape[1],
        }

    def __len__(self):
        return len(self.videos)

    def _load_vid(self, p):
        files = os.listdir(p)
        files = list(filter(lambda file: file.find(".jpg") != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [
            cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array
        ]
        array = np.stack(array, axis=0).astype(np.float32)

        return array

    def _load_anno(self, name):
        with open(name, "r") as f:
            lines = [line.strip().split(" ") for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ["SIL", "SP"], txt))
        return MyDataset.txt2arr(" ".join(txt).upper(), 1)

    def _load_coords(self, name):
        # obtained from the resized image in the lip coordinate extraction
        img_width = 600
        img_height = 500
        with open(name, "r") as f:
            coords_data = json.load(f)

        coords = []
        for frame in sorted(coords_data.keys(), key=int):
            frame_coords = coords_data[frame]

            # Normalize the coordinates
            normalized_coords = []
            for x, y in zip(frame_coords[0], frame_coords[1]):
                normalized_x = x / img_width
                normalized_y = y / img_height
                normalized_coords.append((normalized_x, normalized_y))

            coords.append(normalized_coords)
        coords_array = np.array(coords, dtype=np.float32)
        return coords_array

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    @staticmethod

    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)

    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if n >= start:
                txt.append(MyDataset.letters[n - start])
        return "".join(txt).strip()

    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if pre != n and n >= start:
                if (
                    len(txt) > 0
                    and txt[-1] == " "
                    and MyDataset.letters[n - start] == " "
                ):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])
            pre = n
        return "".join(txt).strip()

    @staticmethod
    def wer(predict, truth):
        word_pairs = [(p[0].split(" "), p[1].split(" ")) for p in zip(predict, truth)]
        wer = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in word_pairs]
        return wer

    @staticmethod
    def cer(predict, truth):
        cer = [
            1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in zip(predict, truth)
        ]
        return cer