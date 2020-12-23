import enum
import torch
import math, logging
from argparse import Namespace  # for type


class Generator(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace) -> None:
        super(Generator, self).__init__()

        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['First'] = torch.nn.Sequential()
        self.layer_Dict['First'].add_module('Conv', Conv1d(
            in_channels= 1,
            out_channels= self.hp.Generator.Residual_Channels,
            kernel_size= 1,
            w_init_gain= 'linear'
            ))
        
        for block_Index in range(self.hp.Generator.ResConvGLU.Blocks):
            for stack_Index in range(self.hp.Generator.ResConvGLU.Stacks_in_Block):
                self.layer_Dict['ResConvGLU_{}_{}'.format(block_Index, stack_Index)] = ResConvGLU(
                    residual_channels= self.hp.Generator.Residual_Channels,
                    gate_channels= self.hp.Generator.ResConvGLU.Gate_Channels,
                    skip_channels= self.hp.Generator.ResConvGLU.Skip_Channels,
                    aux_channels= self.hp.Sound.Mel_Dim + 2,    # Mels + Silences + Pitches
                    kernel_size= self.hp.Generator.ResConvGLU.Kernel_Size,
                    dilation= 2 ** stack_Index,
                    dropout= self.hp.Generator.ResConvGLU.Dropout_Rate,
                    bias= True
                    )

        self.layer_Dict['Last'] = torch.nn.Sequential()
        self.layer_Dict['Last'].add_module('ReLU_0', torch.nn.ReLU(inplace= True))
        self.layer_Dict['Last'].add_module('Conv_0', Conv1d(
            in_channels= self.hp.Generator.ResConvGLU.Skip_Channels,
            out_channels= self.hp.Generator.ResConvGLU.Skip_Channels,
            kernel_size= 1,
            bias= True,
            w_init_gain= 'relu'
            ))
        self.layer_Dict['Last'].add_module('ReLU_1', torch.nn.ReLU(inplace= True))
        self.layer_Dict['Last'].add_module('Conv_1', Conv1d(
            in_channels= self.hp.Generator.ResConvGLU.Skip_Channels,
            out_channels= 1,
            kernel_size= 1,
            bias= True,
            w_init_gain= 'linear'
            ))  #[Batch, 1, Time]

        self.layer_Dict['Upsample'] = UpsampleNet(self.hp)

        self.apply_weight_norm()
        
    def forward(
        self,
        x: torch.FloatTensor,
        mels: torch.FloatTensor,
        silences: torch.LongTensor,
        pitches: torch.FloatTensor
        ) -> torch.FloatTensor:
        '''
        x: [Batch, Time]
        mels: [Batch, Mel, Mel_Time]
        silences: [Batch, Mel_Time]
        pitches: [Batch, Mel_Time]
        '''
        auxs = torch.cat([
            mels,
            silences.unsqueeze(dim= 1).float(),
            pitches.unsqueeze(dim= 1)
            ], axis= 1)

        auxs = self.layer_Dict['Upsample'](auxs)    # [Batch, Mel, Time]
        x = self.layer_Dict['First'](x.unsqueeze(1))    # [Batch, Res, Time]

        skips = 0
        for block_Index in range(self.hp.Generator.ResConvGLU.Blocks):
            for stack_Index in range(self.hp.Generator.ResConvGLU.Stacks_in_Block):
                x, new_Skips = self.layer_Dict['ResConvGLU_{}_{}'.format(block_Index, stack_Index)](x, auxs)
                skips += new_Skips
        skips *= math.sqrt(1.0 / (self.hp.Generator.ResConvGLU.Blocks * self.hp.Generator.ResConvGLU.Stacks_in_Block))

        logits = self.layer_Dict['Last'](skips).squeeze(dim= 1) #[Batch, Time]

        return logits

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
                
        self.apply(_apply_weight_norm)


class Discriminators(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace) -> None:
        super(Discriminators, self).__init__()
        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()

        for index, sampling_Size in enumerate(self.hp.Discriminator.Sampling_Sizes):
            self.layer_Dict['Discriminator_{}'.format(index)] = Discriminator(
                stacks= self.hp.Discriminator.Stacks,
                channels= self.hp.Discriminator.Channels,
                kernel_size= self.hp.Discriminator.Kernel_Size,
                sampling_size= sampling_Size
                )

    def forward(
        self,
        x: torch.FloatTensor
        ):
        '''
        x: [Batch, Time]
        '''
        return [
            self.layer_Dict['Discriminator_{}'.format(index)](x)
            for index in range(len(self.hp.Discriminator.Sampling_Sizes))
            ]

class Discriminator(torch.nn.Module):
    def __init__(
        self,
        stacks: int,
        channels: int,
        kernel_size: int,
        sampling_size: int
        ) -> None:
        super(Discriminator, self).__init__()
        self.sampling_size = sampling_size

        self.layer = torch.nn.Sequential()

        previous_Channels = 1
        for index in range(stacks - 1):
            dilation = max(1, index)
            padding = (kernel_size - 1) // 2 * dilation
            self.layer.add_module('Conv_{}'.format(index), Conv1d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_size,
                padding= padding,
                dilation= dilation,
                w_init_gain= 'leaky_relu'
                ))
            self.layer.add_module('LeakyReLU_{}'.format(index),  torch.nn.LeakyReLU(
                negative_slope= 0.2,
                inplace= True
                ))
            previous_Channels = channels

        self.layer.add_module('Last', Conv1d(
            in_channels= previous_Channels,
            out_channels= 1,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            bias= True
            ))

        self.apply_weight_norm()

    def forward(
        self,
        x: torch.FloatTensor
        ):
        '''
        x: [Batch, Time]
        '''
        x = self.layer(x.unsqueeze(dim= 1)).squeeze(1)
        if x.size(1) == self.sampling_size:
            return x

        offset = torch.randint(0, x.size(1) - self.sampling_size, (1,))

        return x[:, offset:offset+self.sampling_size]

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class UpsampleNet(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super(UpsampleNet, self).__init__()
        self.hp = hyper_parameters

        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['First'] = Conv1d(
            in_channels= self.hp.Sound.Mel_Dim + 2, # Mels + Silences + Pitches
            out_channels= self.hp.Sound.Mel_Dim + 2,
            kernel_size= self.hp.Generator.Upsample.Pad * 2 + 1,
            bias= False,
            w_init_gain= 'linear'
            )   # [Batch, Aux_dim, Time]

        for index, scale in enumerate(self.hp.Generator.Upsample.Scales):
            self.layer_Dict['Stretch_{}'.format(index)] = Stretch2d(scale, 1, mode='nearest')  # [Batch, 1, Aux_dim, Scaled_Time]
            self.layer_Dict['Conv2d_{}'.format(index)] = Conv2d(
                in_channels= 1,
                out_channels= 1,
                kernel_size= (1, scale * 2 + 1),
                padding= (0, scale),
                bias= False,
                w_init_gain= 'linear'
                )   # [Batch, 1, Aux_dim, Scaled_Time]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        '''
        x: [Batch, Mel, Time]
        '''
        x = self.layer_Dict['First'](x).unsqueeze(dim= 1)   # [Batch, 1, Aux_dim, Time]
        for index in range(len(self.hp.Generator.Upsample.Scales)):
            x = self.layer_Dict['Stretch_{}'.format(index)](x)  # [Batch, 1, Aux_dim, Scaled_Time]
            x = self.layer_Dict['Conv2d_{}'.format(index)](x)   # [Batch, 1, Aux_dim, Scaled_Time]

        return x.squeeze(dim= 1)

class ResConvGLU(torch.nn.Module):
    def __init__(
        self,
        residual_channels,
        gate_channels,
        skip_channels,
        aux_channels,
        kernel_size,
        dilation= 1,
        dropout= 0.0,
        bias= True
        ):
        super(ResConvGLU, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Conv1d'] = torch.nn.Sequential()
        self.layer_Dict['Conv1d'].add_module('Dropout', torch.nn.Dropout(p= dropout))
        self.layer_Dict['Conv1d'].add_module('Conv1d', Conv1d(
            in_channels= residual_channels,
            out_channels= gate_channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2 * dilation,
            dilation= dilation,
            bias= bias,
            w_init_gain= ['tanh', 'sigmoid']
            ))

        self.layer_Dict['Aux'] = Conv1d(
            in_channels= aux_channels,
            out_channels= gate_channels,
            kernel_size= 1,
            bias= False,
            w_init_gain= ['tanh', 'sigmoid']
            )

        self.layer_Dict['Out'] = Conv1d(
            in_channels= gate_channels // 2,
            out_channels= residual_channels,
            kernel_size= 1,
            bias= bias,
            w_init_gain= 'linear'
            )

        self.layer_Dict['Skip'] = Conv1d(
            in_channels= gate_channels // 2,
            out_channels= skip_channels,
            kernel_size= 1,
            bias= bias,
            w_init_gain= 'linear'
            )

    def forward(self, audios, auxs):
        residuals = audios

        audios = self.layer_Dict['Conv1d'](audios)
        audios_Tanh, audios_Sigmoid = audios.chunk(2, dim= 1)

        auxs = self.layer_Dict['Aux'](auxs)
        auxs_Tanh, auxs_Sigmoid = auxs.chunk(2, dim= 1)

        audios_Tanh = torch.tanh(audios_Tanh + auxs_Tanh)
        audios_Sigmoid = torch.sigmoid(audios_Sigmoid + auxs_Sigmoid)
        audios = audios_Tanh * audios_Sigmoid 

        outs = (self.layer_Dict['Out'](audios) + residuals) * math.sqrt(0.5)
        skips = self.layer_Dict['Skip'](audios)

        return outs, skips


class Conv1d(torch.nn.Conv1d):
    def __init__(self, w_init_gain= 'relu', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        gains = self.w_init_gain
        if isinstance(gains, str) or isinstance(gains, float):
            gains = [gains]
        
        weights = torch.chunk(self.weight, len(gains), dim= 0)
        for gain, weight in zip(gains, weights):
            if gain == 'zero':
                torch.nn.init.zeros_(weight)
            elif gain in ['relu', 'leaky_relu']:
                torch.nn.init.kaiming_uniform_(weight, nonlinearity= gain)
            else:
                if type(gain) == str:
                    gain = torch.nn.init.calculate_gain(gain)
                torch.nn.init.xavier_uniform_(weight, gain= gain)

        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Conv2d(torch.nn.Conv2d):
    def __init__(self, w_init_gain= 'relu', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Stretch2d(torch.nn.Module):
    def __init__(self, x_scale, y_scale, mode= 'nearest'):
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode= mode

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            scale_factor=(self.y_scale, self.x_scale),
            mode= self.mode
            )



class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes,
        shift_lengths,
        win_lengths,
        window= torch.hann_window
        ):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.layer_Dict = torch.nn.ModuleDict()

        for index, (fft_Size, shift_Length, win_Length) in enumerate(zip(
            fft_sizes,
            shift_lengths,
            win_lengths
            )):            
            self.layer_Dict['STFTLoss_{}'.format(index)] = STFTLoss(
                fft_size= fft_Size,
                shift_length= shift_Length,
                win_length= win_Length,
                window= window
                )

    def forward(self, x, y):
        spectral_Convergence_Loss = 0.0
        magnitude_Loss = 0.0
        for layer in self.layer_Dict.values():
            new_Spectral_Convergence_Loss, new_Magnitude_Loss = layer(x, y)
            spectral_Convergence_Loss += new_Spectral_Convergence_Loss
            magnitude_Loss += new_Magnitude_Loss

        spectral_Convergence_Loss /= len(self.layer_Dict)
        magnitude_Loss /= len(self.layer_Dict)

        return spectral_Convergence_Loss, magnitude_Loss

class STFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_size,
        shift_length,
        win_length,
        window= torch.hann_window
        ):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_length = shift_length
        self.win_length = win_length
        self.window = window
        
        self.l1_loss_layer = torch.nn.L1Loss()

    def forward(self, x, y):
        x_Magnitute = self.STFT(x)
        y_Magnitute = self.STFT(y)

        spectral_Convergence_Loss = self.SpectralConvergenceLoss(x_Magnitute, y_Magnitute)
        magnitude_Loss = self.LogSTFTMagnitudeLoss(x_Magnitute, y_Magnitute)
        
        return spectral_Convergence_Loss, magnitude_Loss

    def STFT(self, x):
        x_STFT = torch.stft(
            input= x,
            n_fft= self.fft_size,
            hop_length= self.shift_length,
            win_length= self.win_length,
            window= self.window(self.win_length).to(x.device)
            )
        reals, imags = x_STFT[..., 0], x_STFT[..., 1]

        return torch.sqrt(torch.clamp(reals ** 2 + imags ** 2, min= 1e-7)).transpose(2, 1)

    def LogSTFTMagnitudeLoss(self, x_magnitude, y_magnitude):
        return self.l1_loss_layer(torch.log(x_magnitude), torch.log(y_magnitude))

    def SpectralConvergenceLoss(self, x_magnitude, y_magnitude):
        return torch.norm(y_magnitude - x_magnitude, p='fro') / torch.norm(y_magnitude, p='fro')



if __name__ == "__main__":
    import yaml
    from Arg_Parser import Recursive_Parse
    hp = Recursive_Parse(yaml.load(
        open('Hyper_Parameters.yaml', encoding='utf-8'),
        Loader=yaml.Loader
        ))

    # from Datasets import Dataset, Collater
    # token_Dict = yaml.load(open(hp.Token_Path), Loader=yaml.Loader)
    # dataset = Dataset(
    #     pattern_path= hp.Train.Train_Pattern.Path,
    #     Metadata_file= hp.Train.Train_Pattern.Metadata_File,
    #     token_dict= token_Dict,
    #     accumulated_dataset_epoch= hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
    #     )
    # collater = Collater(
    #     token_dict= token_Dict,
    #     max_mel_length= hp.Train.Max_Mel_Length
    #     )
    # dataLoader = torch.utils.data.DataLoader(
    #     dataset= dataset,
    #     collate_fn= collater,
    #     sampler= torch.utils.data.RandomSampler(dataset),
    #     batch_size= hp.Train.Batch_Size,
    #     num_workers= hp.Train.Num_Workers,
    #     pin_memory= True
    #     )
    
    # durations, tokens, notes, mels, mel_Lengths = next(iter(dataLoader))

    generator = Generator(hp)
    discriminators = Discriminators(hp)
    x = generator(
        x= torch.randn(4, 480 * 100),
        mels= torch.randn(4, 80, 204),
        silences= torch.randint(0,2,size=(4, 204)),
        pitches= torch.randn(4, 204)
        )
    x = discriminators(x)
    
    print([q.size() for q in x])
