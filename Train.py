import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, sys, argparse, math
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch.multiprocessing as mp

from Modules import Generator, Discriminators, MultiResolutionSTFTLoss
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Radam import RAdam
from Noam_Scheduler import Modified_Noam_Scheduler
from Logger import Logger
from Arg_Parser import Recursive_Parse

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

try:
    from apex import amp
    is_AMP_Exist = True
except:
    logging.info('There is no apex modules in the environment. Mixed precision does not work.')
    is_AMP_Exist = False

class Trainer:
    def __init__(self, hp_path, steps= 0, gpu_id= 0):
        self.hp_Path = hp_path
        self.gpu_id = gpu_id
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_Path, encoding='utf-8'),
            Loader=yaml.Loader
            ))
        if not is_AMP_Exist:
            self.hp.Use_Mixed_Precision = False

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(gpu_id))
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(0)

        self.steps = steps

        self.Datset_Generate()
        self.Model_Generate()

        self.scalar_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        self.writer_Dict = {
            'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
            'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
            }
        
        self.Load_Checkpoint()

    def Datset_Generate(self):
        train_Dataset = Dataset(
            pattern_path= self.hp.Train.Train_Pattern.Path,
            Metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            use_cache = self.hp.Train.Use_Pattern_Cache
            )
        eval_Dataset = Dataset(
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            Metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            use_cache = self.hp.Train.Use_Pattern_Cache
            )
        inference_Dataset = Inference_Dataset(
            pattern_paths= 'Inference_Wav_for_Training.txt',
            use_cache= self.hp.Train.Use_Pattern_Cache
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(train_Dataset.base_Length))
            logging.info('The number of development patterns = {}.'.format(eval_Dataset.base_Length))
            logging.info('The number of inference patterns = {}.'.format(len(inference_Dataset)))

        collater = Collater(
            wav_length= self.hp.Train.Wav_Length,
            frame_shift= self.hp.Sound.Frame_Shift,
            upsample_pad= self.hp.Generator.Upsample.Pad,
            )
        inference_Collater = Inference_Collater(
            wav_length= self.hp.Train.Wav_Length,
            frame_shift= self.hp.Sound.Frame_Shift,
            upsample_pad= self.hp.Generator.Upsample.Pad,
            max_abs_mel= self.hp.Sound.Max_Abs_Mel
            )

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            sampler= torch.utils.data.DistributedSampler(train_Dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_Dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_Dataset,
            sampler= torch.utils.data.RandomSampler(eval_Dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_Dataset,
            sampler= torch.utils.data.SequentialSampler(inference_Dataset),
            collate_fn= inference_Collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        if self.hp.Use_Multi_GPU:
            self.model_Dict = {
                'Generator': torch.nn.parallel.DistributedDataParallel(
                    Generator(self.hp).to(self.device),
                    device_ids=[self.gpu_id]
                    ),
                'Discriminator': torch.nn.parallel.DistributedDataParallel(
                    Discriminators(self.hp).to(self.device),
                    device_ids=[self.gpu_id]
                    )
                }
        else:
            self.model_Dict = {
                'Generator': Generator(self.hp).to(self.device),
                'Discriminator': Discriminators(self.hp).to(self.device)
                }

        self.criterion_Dict = {
            'STFT': MultiResolutionSTFTLoss(
                fft_sizes= self.hp.STFT_Loss_Resolution.FFT_Sizes,
                shift_lengths= self.hp.STFT_Loss_Resolution.Shfit_Lengths,
                win_lengths= self.hp.STFT_Loss_Resolution.Win_Lengths,
                ).to(self.device),
            'Mean_Squared_Error': torch.nn.MSELoss().to(self.device)
            }
        self.optimizer_Dict = {
            'Generator': RAdam(
                params= self.model_Dict['Generator'].parameters(),
                lr= self.hp.Train.Learning_Rate.Generator.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon,
                weight_decay= self.hp.Train.Weight_Decay
                ),
            'Discriminator': RAdam(
                params= self.model_Dict['Discriminator'].parameters(),
                lr= self.hp.Train.Learning_Rate.Discriminator.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon,
                weight_decay= self.hp.Train.Weight_Decay
                )
            }
        
        self.scheduler_Dict = {
            'Generator': Modified_Noam_Scheduler(
                optimizer= self.optimizer_Dict['Generator'],
                base= self.hp.Train.Learning_Rate.Generator.Base
                ),
            'Discriminator': Modified_Noam_Scheduler(
                optimizer= self.optimizer_Dict['Discriminator'],
                base= self.hp.Train.Learning_Rate.Discriminator.Base
                )
            }
        
        if self.hp.Use_Mixed_Precision:
            amp_Wrapped = amp.initialize(
                models=[self.model_Dict['Generator'], self.model_Dict['Discriminator']],
                optimizers=[self.optimizer_Dict['Generator'], self.optimizer_Dict['Discriminator']]
                )
            self.model_Dict['Generator'], self.model_Dict['Discriminator'] = amp_Wrapped[0]
            self.optimizer_Dict['Generator'], self.optimizer_Dict['Discriminator'] = amp_Wrapped[1]

        if self.gpu_id == 0:
            logging.info('#' * 100)
            logging.info('Generator structure')
            logging.info(self.model_Dict['Generator'])
            logging.info('#' * 100)
            logging.info('Discriminator structure')
            logging.info(self.model_Dict['Discriminator'])

    def Train_Step(self, noises, mels, silences, pitches, audios):
        loss_Dict = {}

        noises = noises.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        silences = silences.to(self.device, non_blocking=True)
        pitches = pitches.to(self.device, non_blocking=True)
        audios = audios.to(self.device, non_blocking=True)

        fakes = self.model_Dict['Generator'](
            x= noises,
            mels= mels,
            silences= silences,
            pitches= pitches
            )

        loss_Dict['Spectral_Convergence'], loss_Dict['Magnitude'] = self.criterion_Dict['STFT'](fakes, audios)
        loss_Dict['Generator'] = loss_Dict['Spectral_Convergence'] + loss_Dict['Magnitude']

        if self.steps >= self.hp.Train.Discriminator_Delay:
            fake_Discriminations = self.model_Dict['Discriminator'](fakes)
            loss_Dict['Adversarial'] = 0.0
            for discrimination in fake_Discriminations:
                loss_Dict['Adversarial'] += self.criterion_Dict['Mean_Squared_Error'](
                    discrimination,
                    discrimination.new_ones(discrimination.size())
                    )
            loss_Dict['Generator'] += loss_Dict['Adversarial']
        
        self.optimizer_Dict['Generator'].zero_grad()
        if self.hp.Use_Mixed_Precision:
            with amp.scale_loss(loss_Dict['Generator'], self.optimizer_Dict['Generator']) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= amp.master_params(self.optimizer_Dict['Generator']),
                max_norm= self.hp.Train.Gradient_Norm
                )
        else:
            loss_Dict['Generator'].backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_Dict['Generator'].parameters(),
                max_norm=  self.hp.Train.Gradient_Norm
                )
        self.optimizer_Dict['Generator'].step()
        self.scheduler_Dict['Generator'].step()

        if self.steps >= self.hp.Train.Discriminator_Delay:
            real_Discriminations = self.model_Dict['Discriminator'](audios)
            fake_Discriminations = self.model_Dict['Discriminator'](fakes.detach())

            loss_Dict['Real'] = 0.0
            for discrimination in real_Discriminations:
                loss_Dict['Real'] += self.criterion_Dict['Mean_Squared_Error'](
                    discrimination,
                    discrimination.new_ones(discrimination.size())
                    )
            loss_Dict['Fake'] = 0.0
            for discrimination in fake_Discriminations:
                loss_Dict['Fake'] += discrimination.mean()
            loss_Dict['Discriminator'] = loss_Dict['Real'] + loss_Dict['Fake']

            self.optimizer_Dict['Discriminator'].zero_grad()
            if self.hp.Use_Mixed_Precision:
                with amp.scale_loss(loss_Dict['Discriminator'], self.optimizer_Dict['Discriminator']) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters= amp.master_params(self.optimizer_Dict['Discriminator']),
                    max_norm= self.hp.Train.Gradient_Norm
                    )
            else:
                loss_Dict['Discriminator'].backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters= self.model_Dict['Discriminator'].parameters(),
                    max_norm= self.hp.Train.Gradient_Norm
                    )
            self.optimizer_Dict['Discriminator'].step()
            self.scheduler_Dict['Discriminator'].step()

        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for noises, mels, silences, pitches, audios in self.dataLoader_Dict['Train']:
            self.Train_Step(noises, mels, silences, pitches, audios)
            
            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0:
                self.scalar_Dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_Dict['Train'].items()
                    }
                self.scalar_Dict['Train']['Learning_Rate/Generator'] = self.scheduler_Dict['Generator'].get_last_lr()
                if self.steps >= self.hp.Train.Discriminator_Delay:
                    self.scalar_Dict['Train']['Learning_Rate/Discriminator'] = self.scheduler_Dict['Discriminator'].get_last_lr()
                self.writer_Dict['Train'].add_scalar_dict(self.scalar_Dict['Train'], self.steps)
                self.scalar_Dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    @torch.no_grad()
    def Evaluation_Step(self, noises, mels, silences, pitches, audios):
        loss_Dict = {}

        noises = noises.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        silences = silences.to(self.device, non_blocking=True)
        pitches = pitches.to(self.device, non_blocking=True)
        audios = audios.to(self.device, non_blocking=True)

        fakes = self.model_Dict['Generator'](
            x= noises,
            mels= mels,
            silences= silences,
            pitches= pitches
            )

        loss_Dict['Spectral_Convergence'], loss_Dict['Magnitude'] = self.criterion_Dict['STFT'](fakes, audios)
        loss_Dict['Generator'] = loss_Dict['Spectral_Convergence'] + loss_Dict['Magnitude']

        if self.steps >= self.hp.Train.Discriminator_Delay:
            fake_Discriminations = self.model_Dict['Discriminator'](fakes)
            loss_Dict['Adversarial'] = 0.0
            for discrimination in fake_Discriminations:
                loss_Dict['Adversarial'] += self.criterion_Dict['Mean_Squared_Error'](
                    discrimination,
                    discrimination.new_ones(discrimination.size())
                    )
            loss_Dict['Generator'] += loss_Dict['Adversarial']
        
        if self.steps >= self.hp.Train.Discriminator_Delay:
            real_Discriminations = self.model_Dict['Discriminator'](audios)
            fake_Discriminations = self.model_Dict['Discriminator'](fakes.detach())

            loss_Dict['Real'] = 0.0
            for discrimination in real_Discriminations:
                loss_Dict['Real'] += self.criterion_Dict['Mean_Squared_Error'](
                    discrimination,
                    discrimination.new_ones(discrimination.size())
                    )
            loss_Dict['Fake'] = 0.0
            for discrimination in fake_Discriminations:
                loss_Dict['Fake'] += discrimination.mean()
            loss_Dict['Discriminator'] = loss_Dict['Real'] + loss_Dict['Fake']

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Evaluation']['Loss/{}'.format(tag)] += loss.cpu()

        return fakes

    def Evaluation_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model_Dict['Generator'].eval()
        self.model_Dict['Discriminator'].eval()

        for step, (noises, mels, silences, pitches, audios) in tqdm(
            enumerate(self.dataLoader_Dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataLoader_Dict['Eval'].dataset) / self.hp.Train.Batch_Size)
            ):
            fakes = self.Evaluation_Step(noises, mels, silences, pitches, audios)

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
            }
        self.writer_Dict['Evaluation'].add_scalar_dict(self.scalar_Dict['Evaluation'], self.steps)
        self.writer_Dict['Evaluation'].add_histogram_model(self.model_Dict['Generator'], self.steps, delete_keywords=['layer_Dict', 'layer', 'module'])
        self.writer_Dict['Evaluation'].add_histogram_model(self.model_Dict['Discriminator'], self.steps, delete_keywords=['layer_Dict', 'layer', 'module'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        image_Dict = {
            'Mel': (mels[-1].cpu().numpy(), None),
            'Silence': (silences[-1].cpu().numpy(), None),
            'Pitch': (pitches[-1].cpu().numpy(), None),
            'Audio/Target': (audios[-1].cpu().numpy(), None),
            'Audio/Prediction': (fakes[-1].cpu().numpy(), None),
            }
        self.writer_Dict['Evaluation'].add_image_dict(image_Dict, self.steps)

        self.model_Dict['Generator'].train()
        self.model_Dict['Discriminator'].train()

    @torch.no_grad()
    def Inference_Step(self, noises, mels, silences, pitches, labels, start_index= 0, tag_step= False):
        noises = noises.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        silences = silences.to(self.device, non_blocking=True)
        pitches = pitches.to(self.device, non_blocking=True)

        fakes = self.model_Dict['Generator'](
            x= noises,
            mels= mels,
            silences= silences,
            pitches= pitches
            )
        
        files = []
        for index, label in enumerate(labels):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append(label)
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav').replace('\\', '/'), exist_ok= True)
        for fake, mel, silence, pitch, label, file in zip(
            fakes.cpu().numpy(),
            mels.cpu().numpy(),
            silences.cpu().numpy(),
            pitches.cpu().numpy(),
            labels,
            files
            ):
            title = 'Label: {}'.format(label)
            new_Figure = plt.figure(figsize=(20, 5 * 4), dpi=100)

            plt.subplot2grid((4, 1), (0, 0))
            plt.plot(fake)
            plt.margins(x= 0)
            plt.title('Audio    {}'.format(title))

            plt.subplot2grid((4, 1), (1, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel    {}'.format(title))

            plt.subplot2grid((4, 1), (2, 0))
            plt.plot(silence)
            plt.margins(x= 0)
            plt.title('Silence    {}'.format(title))

            plt.subplot2grid((4, 1), (3, 0))
            plt.plot(pitch)
            plt.margins(x= 0)
            plt.title('Pitch    {}'.format(title))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_Figure)
            
            np.save(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel', file).replace('\\', '/'),
                mel.T,
                allow_pickle= False
                )                
            
            wavfile.write(
                filename= os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav', '{}.wav'.format(file)).replace('\\', '/'),
                data= (np.clip(fake, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
                rate= self.hp.Sound.Sample_Rate
                )
            
    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start inference in GPU {}.'.format(self.steps, self.gpu_id))

        self.model_Dict['Generator'].eval()

        for step, (noises, mels, silences, pitches, labels) in tqdm(
            enumerate(self.dataLoader_Dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / (self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))
            ):
            self.Inference_Step(noises, mels, silences, pitches, labels, start_index= step * (self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))

        self.model_Dict['Generator'].train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_Dict = torch.load(path, map_location= 'cpu')
        
        if self.hp.Use_Multi_GPU:
            self.model_Dict['Generator'].module.load_state_dict(state_Dict['Generator']['Model'])
            self.model_Dict['Discriminator'].module.load_state_dict(state_Dict['Discriminator']['Model'])
        else:
            self.model_Dict['Generator'].load_state_dict(state_Dict['Generator']['Model'])
            self.model_Dict['Discriminator'].load_state_dict(state_Dict['Discriminator']['Model'])

        self.optimizer_Dict['Generator'].load_state_dict(state_Dict['Generator']['Optimizer'])
        self.optimizer_Dict['Discriminator'].load_state_dict(state_Dict['Discriminator']['Optimizer'])

        self.scheduler_Dict['Generator'].load_state_dict(state_Dict['Generator']['Scheduler'])
        self.scheduler_Dict['Discriminator'].load_state_dict(state_Dict['Discriminator']['Scheduler'])

        self.steps = state_Dict['Steps']

        if self.hp.Use_Mixed_Precision:
            if not 'AMP' in state_Dict.keys():
                logging.info('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)

        state_Dict = {
            'Generator': {
                'Model': self.model_Dict['Generator'].module.state_dict() if self.hp.Use_Multi_GPU else self.model_Dict['Generator'].state_dict(),
                'Optimizer': self.optimizer_Dict['Generator'].state_dict(),
                'Scheduler': self.scheduler_Dict['Generator'].state_dict(),
                },
            'Discriminator': {
                'Model': self.model_Dict['Discriminator'].module.state_dict() if self.hp.Use_Multi_GPU else self.model_Dict['Discriminator'].state_dict(),
                'Optimizer': self.optimizer_Dict['Discriminator'].state_dict(),
                'Scheduler': self.scheduler_Dict['Discriminator'].state_dict(),
                },
            'Steps': self.steps
            }
        if self.hp.Use_Mixed_Precision:
            state_Dict['AMP'] = amp.state_dict()

        torch.save(
            state_Dict,
            os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

    def Train(self):
        hp_Path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_Path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_Path, hp_Path)

        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')


def Worker(gpu, hp_path, steps):
    torch.distributed.init_process_group(
        backend= 'nccl',
        init_method='tcp://127.0.0.1:54321',
        world_size= torch.cuda.device_count(),
        rank= gpu
        )

    new_Trainer = Trainer(hp_path= hp_path, steps= steps, gpu_id= gpu)
    new_Trainer.Train()

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        mp.spawn(
            Worker,
            nprocs= torch.cuda.device_count(),
            args= (args.hyper_parameters, args.steps)
            )
    else:
        new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps, gpu_id= 0)
        new_Trainer.Train()