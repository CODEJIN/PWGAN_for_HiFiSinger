import torch
import numpy as np
import yaml, os, sys, argparse, time, importlib, math
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Modules import Generator
from Arg_Parser import Recursive_Parse

def Trace(
    hp_path: str,
    checkpoint_path: str,
    export_path: str='./Trace/Vocoder.pts'
    ):    
    hp = Recursive_Parse(yaml.load(
        open(hp_path, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    model = Generator(hyper_parameters= hp)

    state_Dict = torch.load(
        checkpoint_path,
        map_location= 'cpu'
        )
    model.load_state_dict(state_Dict['Generator']['Model'])
    model.remove_weight_norm()
    model.eval()

    for param in model.parameters(): 
        param.requires_grad = False

    x= torch.randn(size=(1, hp.Sound.Frame_Shift * 10))
    mels= torch.randn(size=(1, 80, 10 + 2*2))
    silences= torch.randint(0, 2, size=(1, 10 + 2*2)).long()
    pitches= torch.randn(size=(1, 10 + 2*2))

    traced_model = torch.jit.trace(model, (x, mels, silences, pitches))
    os.makedirs(os.path.dirname(export_path), exist_ok= True)
    traced_model.save(export_path)


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-c', '--checkpoint_path', required= True, type= str)
    argParser.add_argument('-e', '--export_path', default= './Trace/Vocoder.pts', type= str)
    argParser.add_argument('-gpu', '--gpu', default= '0', type= str)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    Trace(
        hp_path= args.hyper_parameters,
        checkpoint_path= args.checkpoint_path,
        export_path= args.export_path
        )

    # python Trace.py -hp .\Hyper_Parameters.yaml -c D:\PWGAN_HiFiSinger.Reuslts\Songs_15\Checkpoint\S_50000.pt -gpu 1