# If there is no duration in pattern dict, you must add the duration information
# Please use 'Get_Duration.py' in Pitchtron repository

import torch
import numpy as np
import pickle, os
from random import randint
from multiprocessing import Manager

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path: str,
        Metadata_file: str,
        accumulated_dataset_epoch: int= 1,
        use_cache: bool= False
        ):
        super(Dataset, self).__init__()
        self.pattern_Path = pattern_path
        self.use_cache = use_cache

        self.metadata_Path = os.path.join(pattern_path, Metadata_file).replace('\\', '/')
        metadata_Dict = pickle.load(open(self.metadata_Path, 'rb'))
        self.patterns = metadata_Dict['File_List']

        self.base_Length = len(self.patterns)
        self.patterns *= accumulated_dataset_epoch
        
        self.cache_Dict = Manager().dict()

    def __getitem__(self, idx: int):
        if (idx % self.base_Length) in self.cache_Dict.keys():
            return self.cache_Dict[self.metadata_Path, idx % self.base_Length]

        path = os.path.join(self.pattern_Path, self.patterns[idx]).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))

        pattern = pattern_Dict['Mel'], pattern_Dict['Silence'], pattern_Dict['Pitch'], pattern_Dict['Audio']
        if self.use_cache:
            self.cache_Dict[self.metadata_Path, idx % self.base_Length] = pattern
        
        return pattern

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_paths: str= 'Inference_Wav_for_Training.txt',
        use_cache: bool= False
        ):
        super(Inference_Dataset, self).__init__()
        self.use_cache = use_cache

        self.patterns = [
            (line.strip().split('\t')[0], line.strip().split('\t')[1])
            for line in open(pattern_paths, 'r', encoding= 'utf-8').readlines()[1:]
            ]

        self.cache_Dict = Manager().dict()

    def __getitem__(self, idx: int):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict['Inference', idx]

        label, path = self.patterns[idx]
        
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = pattern_Dict['Mel'], pattern_Dict['Silence'], pattern_Dict['Pitch'], label

        if self.use_cache:
            self.cache_Dict['Inference', idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        wav_length: int,
        frame_shift: int,
        upsample_pad: int
        ):
        self.wav_Length = wav_length
        self.frame_Shift = frame_shift
        self.mel_Length = wav_length // frame_shift
        self.upsample_Pad = upsample_pad

    def __call__(self, batch: list):
        mels, silences, pitches, audios = zip(*batch)
        mels, silences, pitches, audios = self.Stack(mels, silences, pitches, audios)

        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Mel_dim, Time]
        silences = torch.FloatTensor(silences)   # [Batch, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]
        audios = torch.FloatTensor(audios)   # [Batch, Time]
        noises = torch.randn(size= audios.size()) # [Batch, Time]

        return noises, mels, silences, pitches, audios

    def Stack(self, mels, silences, pitches, audios):
        mel_List, silence_List, pitch_List, audio_List = [], [], [], []
        for mel, silence, pitch, audio in zip(mels, silences, pitches, audios):
            mel_Pad = max(0, self.mel_Length + 2 * self.upsample_Pad - mel.shape[0])
            audio_Pad = max(0, self.wav_Length + 2 * self.upsample_Pad * self.frame_Shift - audio.shape[0])
            mel = np.pad(
                mel,
                [[int(np.floor(mel_Pad / 2)), int(np.ceil(mel_Pad / 2))], [0, 0]],
                mode= 'reflect'
                )
            silence = np.pad(
                silence,
                [int(np.floor(mel_Pad / 2)), int(np.ceil(mel_Pad / 2))],
                mode= 'reflect'
                )
            pitch = np.pad(
                pitch,
                [int(np.floor(mel_Pad / 2)), int(np.ceil(mel_Pad / 2))],
                mode= 'reflect'
                )
            audio = np.pad(
                audio,
                [int(np.floor(audio_Pad / 2)), int(np.ceil(audio_Pad / 2))],
                mode= 'reflect'
                )

            mel_Offset = np.random.randint(self.upsample_Pad, max(mel.shape[0] - (self.mel_Length + self.upsample_Pad), self.upsample_Pad + 1))
            audio_Offset = mel_Offset * self.frame_Shift
            mel = mel[mel_Offset - self.upsample_Pad:mel_Offset + self.mel_Length + self.upsample_Pad]
            silence = silence[mel_Offset - self.upsample_Pad:mel_Offset + self.mel_Length + self.upsample_Pad]
            pitch = pitch[mel_Offset - self.upsample_Pad:mel_Offset + self.mel_Length + self.upsample_Pad]
            audio = audio[audio_Offset:audio_Offset + self.wav_Length]

            mel_List.append(mel)
            silence_List.append(silence)
            pitch_List.append(pitch)
            audio_List.append(audio)

        return np.stack(mel_List, axis= 0), np.stack(silence_List, axis= 0), np.stack(pitch_List, axis= 0), np.stack(audio_List, axis= 0)

class Inference_Collater:
    def __init__(
        self,
        wav_length: int,
        frame_shift: int,
        upsample_pad: int,
        max_abs_mel: float
        ):
        self.wav_Length = wav_length
        self.frame_Shift = frame_shift
        self.mel_Length = wav_length // frame_shift
        self.upsample_Pad = upsample_pad
        self.max_Abs_Mel = max_abs_mel
         
    def __call__(self, batch: list):
        max_Mel_Length = max([mel.shape[0] for mel, _, _, _ in batch])

        mels, silences, pitches, labels = [], [], [], []
        for mel, silence, pitch, label in batch:
            mel = np.pad(
                mel,
                pad_width=[[self.upsample_Pad, max_Mel_Length - mel.shape[0] + self.upsample_Pad], [0, 0]],
                constant_values= -self.max_Abs_Mel
                )
            silence = np.pad(
                silence,
                pad_width=[self.upsample_Pad, max_Mel_Length - silence.shape[0] + self.upsample_Pad],
                constant_values= 0
                )
            pitch = np.pad(
                pitch,
                pad_width=[self.upsample_Pad, max_Mel_Length - pitch.shape[0] + self.upsample_Pad],
                constant_values= 0
                )
            
            mels.append(mel)
            silences.append(silence)
            pitches.append(pitch)
            labels.append(label)
            
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        silences = torch.FloatTensor(np.stack(silences, axis= 0))   # [Batch, Time]
        pitches = torch.FloatTensor(np.stack(pitches, axis= 0))     # [Batch, Time]
        noises = torch.randn(size= (mels.size(0), max_Mel_Length * self.frame_Shift))   # [Batch, Time]
        
        return noises, mels, silences, pitches, labels