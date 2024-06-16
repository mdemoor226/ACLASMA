import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.fft as fft
from math import log
import torchaudio.functional as F_audio
from functools import reduce
from operator import __add__
from dataset.data_stats import get_stats
import numpy as np
import os

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


#https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6 - https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/6
class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,[(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)

class Conv1dSamePadding(nn.Conv1d):
    def __init__(self,*args,**kwargs):
        super(Conv1dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_1d = nn.ZeroPad1d(reduce(__add__,[(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_1d(input), self.weight, self.bias)


class ChannelAttention(nn.Module):
    def __init__(self, planes, r_factor=6):
        super(ChannelAttention, self).__init__()
        #Channel-wise Attention...
        self.gmp = nn.AdaptiveMaxPool2d((1,1))
        self.fc1 = nn.Linear(planes, planes // r_factor)
        self.elu = nn.ELU(inplace=True)
        self.fc2 = nn.Linear(planes // r_factor, planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, In):
        batch = In.shape[0]
        x = self.gmp(In).permute(0,2,3,1)
        x = self.fc1(torch.flatten(x, start_dim=1))
        x = self.elu(x)
        x = self.fc2(x)
        out = self.sigmoid(x).view(batch,-1,1,1)
        return out

class TemporalAttention(nn.Module):
    def __init__(self, planes, k=3, r_factor=6):
        super(TemporalAttention, self).__init__()
        #Temporal-wise Attention...
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        #print(planes)
        self.conv1d = nn.Conv1d(planes, planes, k, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, In):
        batch = In.shape[0]
        x = self.gap(In.permute(0,2,1,3))
        #print(x.shape)
        #input()
        x = self.conv1d(x.squeeze(dim=-1)) #Note: I was unable to just use x.squeeze(). It kept appending a batch dim in the conv1d operation. This might be a PyTorch Bug?
        out = self.sigmoid(x).view(batch,1,-1,1)
        return out

class FrequencyAttention(nn.Module):
    def __init__(self, planes, r_factor=6):
        super(FrequencyAttention, self).__init__()
        #Frequency-wise Attention...
        self.gmp = nn.AdaptiveMaxPool2d((1,1))
        self.fc1 = nn.Linear(planes, planes // r_factor)
        self.elu = nn.ELU(inplace=True)
        self.fc2 = nn.Linear(planes // r_factor, planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, In):
        batch = In.shape[0]
        x = self.gmp(In.permute(0,3,1,2))
        x = self.fc1(torch.flatten(x, start_dim=1))
        x = self.elu(x)
        x = self.fc2(x)
        out = self.sigmoid(x).view(batch,1,1,-1)
        return out


class MDAM(nn.Module):
    def __init__(self, cplanes, tplanes, fplanes, r_factor=6):
        super(MDAM, self).__init__()
        self.CAtt = ChannelAttention(cplanes, r_factor=r_factor)
        self.TAtt = TemporalAttention(tplanes, k=3, r_factor=r_factor) 
        self.FAtt = FrequencyAttention(fplanes, r_factor=r_factor) 
        self.alpha = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=True) 
        self.beta = torch.nn.Parameter(torch.Tensor([0.75]), requires_grad=True) 
        self.gamma = torch.nn.Parameter(torch.Tensor([0.75]), requires_grad=True) 
    
    def forward(self, In):
        #import code
        #code.interact(local=locals())
        #Input Shape In: (B,C,Frames,NumSamples) ---> (B,C,Time,Frequency)
        identity = In
        CAtt = self.CAtt(In) #Out Shape: (B,C,1,1)
        TAtt = self.TAtt(In) #Out Shape: (B,1,Time,1)
        FAtt = self.FAtt(In) #Out Shape: (B,1,1,Frequency)
        
        #Fuse into attentional spectral maps
        Sc = identity*CAtt
        St = identity*TAtt
        Sf = identity*FAtt
        
        #Merge Together
        Normalization = self.alpha + self.beta + self.gamma
        Alpha = self.alpha / Normalization
        Beta = self.beta / Normalization
        Gamma = self.gamma / Normalization
        Output = Alpha*Sc + Beta*St + Gamma*Sf
        return Output


class KWUpperBlock(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    #expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        #downsample: Optional[nn.Module] = None,
        #groups: int = 1,
        #base_width: int = 64,
        dilation: int = 1,
        mdam: bool = True,
        APlanes: list = [16,16,16],
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super(KWUpperBlock, self).__init__()
        self.bn1 = norm_layer(inplanes)
        #self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.conv1 = Conv2dSamePadding(inplanes, planes, kernel_size=3, stride=2, bias=False, dilation=dilation) 
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)#, stride, groups, dilation)
        self.maxpoolpad = nn.ZeroPad2d(reduce(__add__,[(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in (2, 2)]))
        self.maxpool = nn.MaxPool2d((2,2)) 
        self.smallconv = conv1x1(inplanes, planes)
        self.bn3 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes, dilation=dilation)# * self.expansion)
        self.bn4 = norm_layer(planes)
        self.conv4 = conv3x3(planes, planes, dilation=dilation)# * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.mdam = mdam
        if mdam:
            self.MDAM = MDAM(cplanes=APlanes[0], tplanes=APlanes[1], fplanes=APlanes[2])
    
    def _pad(self, x):
        _,_,h,w = x.shape
        return F.pad(x, (0, 0, 1, 1)) #I think? Fix this later if necessary. #((Hin - 1)*Stride - Hin + Kernel_Size) / 2 ---> Kernel = 2, Stride = 2: Hin / 2

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(x)
        identity = x
        out = self.relu(x)
        
        #Use what works best
        out = self.conv1(out) #Needs padding too I think...
        #out = self.conv1(self._pad(out))
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        #n,c,h,w = identity.shape #Correct?
        #skip = self.maxpool(self._pad(identity))
        skip = self.maxpool(self.maxpoolpad(identity))
        skip = self.smallconv(skip)
        out = self.bn3(out + skip)
        skip2 = out
        output = self.relu(out)
        output = self.conv3(output)
        output = self.bn4(output)
        output = self.relu(output)
        output = self.conv4(output)
        if self.mdam:
            output = self.MDAM(output)
        return output + skip2

class KWLowerBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        #downsample: Optional[nn.Module] = None,
        #groups: int = 1,
        #base_width: int = 64,
        dilation: int = 1,
        mdam: bool = True,
        APlanes: list = [16,16,16],
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super(KWLowerBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)#, stride, groups, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes, dilation=dilation)# * self.expansion)
        self.bn3 = norm_layer(planes)
        self.conv4 = conv3x3(planes, planes, dilation=dilation)# * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.mdam = mdam
        if mdam:
            self.MDAM = MDAM(cplanes=APlanes[0], tplanes=APlanes[1], fplanes=APlanes[2])
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(x)
        
        out = self.conv1(out)
        out = self.bn1(out) # Wilkinghoff Model did out = conv + relu; skip = bn(identity) --> bn(out + skip). I think he messed up so I changed it to what it is now.
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out + identity)
        
        skip = out
        output = self.relu(out)
        output = self.conv3(output)
        output = self.bn3(output) #Wilkinghoff Model did relu then Batch Normalization here. I switched it around.
        output = self.relu(output)
        output = self.conv4(output)
        if self.mdam:
            output = self.MDAM(output)
        return output + skip


class Spectrogram(object):
    """
    Compute magnitude spectrograms.
    https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06 : <--- Check this out?
    """

    def __init__(self, sample_rate, fft_size, hop_size, device, f_min=0.0, f_max=None, center=False, n_mels=128, log_mel=False, **kwargs): #Num. of default n_mels/mel_bins changed from 20 to 128
        super(Spectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.n_mels = n_mels
        self.hop_size = hop_size
        self.device = device
        self.center = center
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.log_mel = log_mel
        self.mel_filterbank = F_audio.melscale_fbanks(
                n_freqs=fft_size // 2 + 1,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate).to(self.device)

    def _power_to_db(self, magnitude, amin=1e-16, top_db=80.0):
        """
        https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
        """
        ref_value = torch.max(magnitude, dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]#)
        log_spec = 10.0 * torch.maximum(amin*torch.ones_like(magnitude), magnitude) / log(10)
        log_spec -= 10.0 * torch.maximum(amin*torch.ones_like(ref_value), ref_value) / log(10)
        log_spec = torch.maximum(log_spec, torch.max(log_spec) - top_db)
        
        return log_spec

    def __call__(self, waveforms):
        spectrograms = torch.stft(waveforms, n_fft=self.fft_size, hop_length=self.hop_size, window=torch.hann_window(self.fft_size, device=self.device), center=self.center, onesided=True, return_complex=True) #Center=False?
        #b?,s,f = spectrograms.shape
        #assert spectrograms.shape[2] == 311
        #assert len(spectrograms.shape) == 3
        magnitude_spectrograms = torch.abs(spectrograms).permute(0,2,1)#.permute(1,0)
        if not self.log_mel:
            return magnitude_spectrograms
        
        #Convert to log-mel scale...
        mel_spectrograms = torch.matmul(magnitude_spectrograms**2, self.mel_filterbank)
        log_mel_spectrograms = self._power_to_db(mel_spectrograms)
        return log_mel_spectrograms
    
    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'center': self.center,
            'n_mels': self.n_mels
        }
        return config

class KWInitialBlock(nn.Module):
    def __init__(
        self,
        melcfg,
        inplanes: int,
        planes: int,
        dilation: int = 1,
        in_shape: int = 160000,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super(KWInitialBlock, self).__init__()
        self.Spec = Spectrogram(melcfg['sample_rate'], melcfg['fft_size'], melcfg['hop_size'], melcfg['device'], 
                                melcfg['f_min'], melcfg['f_max'], n_mels=melcfg['mel_bins'], log_mel=melcfg['log_mel'])
        #The input value for BatchNorm1d below was determined by debugging. Add a method here to formally calculate the value. 
        self.SpecBN = nn.BatchNorm1d(inplanes)#I believe the normalization must be done across the "frames" dimension...
        
        self.inconv = Conv2dSamePadding(1, planes, kernel_size=7, stride=2, bias=False, dilation=dilation)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        #self.inconv = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn = norm_layer(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_shape = in_shape
    
    def forward(self, In: Tensor) -> Tensor:
        #print(In.dtype)
        # magnitude
        batch = In.shape[0]
        x = self.Spec(In.view(batch,self.in_shape,))
        #x = x - torch.mean(x, dim=1, keepdim=True) #I think this is correct ... now, I believe
        #print(x.dtype)
        x = self.SpecBN(x)
        x = self.inconv(x.unsqueeze(1))#.unsqueeze(1) #Try Both of These? May have to edit the arguments in the constructor.
        x = self.bn(x)
        x = self.relu(x)
        out = self.maxpool(x)
        return out


class KWFFT(nn.Module):
    def __init__(
        self,
        planes: int,
        fc_inplanes: int,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm1d
    ) -> None:
        super(KWFFT, self).__init__()
        #Consider setting the bias terms to True here (and see what happens)
        self.conv1 = Conv1dSamePadding(1, planes, kernel_size=256, stride=64, bias=False, dilation=dilation)
        self.convbn1 = norm_layer(planes)
        self.conv2 = Conv1dSamePadding(planes, planes, kernel_size=64, stride=32, bias=False, dilation=dilation)
        self.convbn2 = norm_layer(planes)
        self.conv3 = Conv1dSamePadding(planes, planes, kernel_size=16, stride=4, bias=False, dilation=dilation)
        self.convbn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(fc_inplanes, planes, bias=False) #1280 number was obtained from technical report... #And it looks like it was right (Update: 2/7/24)
        #self.fc1 = nn.Linear(1280, planes, bias=False) #1280 number was obtained from technical report... #And it looks like it was right (Update: 2/7/24)
        self.bn1 = norm_layer(planes)
        self.fc2 = nn.Linear(planes, planes, bias=False)
        self.bn2 = norm_layer(planes)
        self.fc3 = nn.Linear(planes, planes, bias=False)
        self.bn3 = norm_layer(planes)
        self.fc4 = nn.Linear(planes, planes, bias=False)
        self.bn4 = norm_layer(planes)
    
    def forward(self, In: Tensor) -> Tensor:
        #Pre-Processing
        batch = In.shape[0]
        #In_complex = torch.view_as_complex(torch.cat([In.unsqueeze(-1), torch.zeros_like(In.unsqueeze(-1))], dim=-1)) #Is the complex cast really necessary? (I don't think so but keeping for now).
        #x = torch.abs(torch.fft.fft(In_complex, dim=1)[:, :(self.raw_dim // 2) + 1]).view(batch,1,-1) #FFT dim is 1 if Batch dimension exists, otherwise 0. Assuming it exists for now.
        x = torch.abs(torch.fft.rfft(In, dim=1)).view(batch,1,-1)

        #DFT-Model
        x = self.relu(self.convbn1(self.conv1(x)))
        x = self.relu(self.convbn2(self.conv2(x)))
        x = self.relu(self.convbn3(self.conv3(x)))
        x = self.relu(self.bn1(self.fc1(torch.flatten(x, start_dim=1)))) #BatchNorm Layers should be okay as long as the tensors are of shape (B, C) (i.e. they contain the batch dimension).
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        out = self.relu(self.bn4(self.fc4(x)))
        return out


class Wilkinghoff(nn.Module):

    def __init__(
        self,
        melcfg,
        out_dims: int = 128,
        mdam: bool = True,
        zero_init_residual: bool = False,
        data_year: str = "./2022-data/",
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
    ) -> None:
        super(Wilkinghoff, self).__init__()
        self._norm_layer = norm_layer
        
        #Log-Mel Network
        self.blocks = 5
        self.dilation = 1
        self.stride = 1
        if data_year == './2023-data/':
            self.inplanes = 561
            self.in_shape = 288000
            factor_dims = 4096 #self.inplanes*(2**(self.blocks - 2)) # <--- Why did I change this/comment it out again?
            self.APlanes = [140, 128]
            self.TPlanes = [70, 35, 18] #These Values were determined by debugging. If desired, update this method to calculate these values directly.
        else:
            self.inplanes = 311
            self.in_shape = 160000
            factor_dims = 2048 #self.inplanes*(2**(self.blocks - 2)) # <--- Why did I change this/comment it out again?
            self.APlanes = [77, 128]
            self.TPlanes = [39, 20, 10] #These Values were determined by debugging. If desired, update this method to calculate these values directly.
        self.FPlanes = [64, 32, 16] #These too...
        self.planes = 16
        self.melcfg = melcfg
        self.mel_network = self._make_mel_network(self.inplanes, self.planes, self.blocks, self.stride, mdam, self.melcfg)
        
        #Log-Mel Embedder
        self.embpoolpad = nn.ZeroPad2d(reduce(__add__,[(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in (1, 10)]))
        self.emb_pool = nn.MaxPool2d((10, 1))#, padding=(5,0))
        self.bnout = nn.BatchNorm1d(factor_dims)
        #self.LNorm = nn.LayerNorm(factor_dims, elementwise_affine=False)
        self.emb_mel = nn.Linear(factor_dims, out_dims, bias=True)
        
        #FFT-Network and Embedder
        self.fft_dims = 128
        self.fft_fcin = 2304 if data_year == './2023-data/' else 1280
        self.fft_network = self._make_fft_network(self.fft_dims, self.fft_fcin)
        self.emb_fft = nn.Linear(self.fft_dims, out_dims, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dSamePadding) or isinstance(m, Conv1dSamePadding):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0)
        
        #Data Normalization/Standardization Stats
        Target_sr = self.melcfg['sample_rate']
        if os.path.isfile(data_year + "saved_data/data_mean.npy"):
            mean = np.load(data_year + "saved_data/data_mean.npy")
            std = np.load(data_year + "saved_data/data_std.npy")
            if data_year == './2022-data/' and mean.shape[0] / Target_sr != 10:
                print("Error, saved dataset statistics are not compatible. Updating now... (Make sure you have 64+ gbs of RAM")
                mean, std = get_stats(Target_sr, data_year)
        else:
            mean, std = get_stats(Target_sr, data_year)
        
        self.data_mean = torch.nn.Parameter(torch.from_numpy(mean), requires_grad=False)
        self.data_std = torch.nn.Parameter(torch.from_numpy(std), requires_grad=False)
        
        print(":)")
        self._initialize_weights()

        #NEEDS EDITING IF GOING TO BE USED
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        #if zero_init_residual:
        #    for m in self.modules():
        #        if isinstance(m, Bottleneck):
        #            nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #        elif isinstance(m, BasicBlock):
        #            nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.emb_mel.weight, mode='fan_out')
        nn.init.constant_(self.emb_mel.bias, 0)
        nn.init.kaiming_normal_(self.emb_fft.weight, mode='fan_out')
        nn.init.constant_(self.emb_fft.bias, 0)
    
    def _make_mel_network(self, init_planes: int, planes: int, blocks: int, stride: int = 1, mdam: bool = True, melcfg=None) -> nn.Sequential:
        norm_layer = self._norm_layer
        
        #1st Block
        layers = []
        layers.append(KWInitialBlock(self.melcfg, init_planes, planes, in_shape=self.in_shape, norm_layer=norm_layer))

        #2nd Block
        APlanes = [planes] + self.APlanes
        layers.append(KWLowerBlock(planes, planes, stride=stride, dilation=self.dilation, mdam=mdam, APlanes=APlanes, norm_layer=norm_layer))
        
        #Blocks 3+
        TPlanes = self.TPlanes
        FPlanes = self.FPlanes
        for i in range(2, blocks):
            in_planes = (2**(i - 2))*planes
            layers.append(KWUpperBlock(in_planes, 2*in_planes, stride=stride, dilation=self.dilation, mdam=mdam, APlanes=[2*in_planes,TPlanes[i-2],FPlanes[i-2]], norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _make_fft_network(self, planes: int, fc_inplanes: int) -> nn.Sequential:
        #Network-Build
        return nn.Sequential(KWFFT(planes, fc_inplanes, norm_layer=nn.BatchNorm1d))

    def _forward_impl(self, In: Tensor) -> Tensor:
        #Standardize input data
        In = (In - self.data_mean) / self.data_std
        
        #Log-Mel Spectrogram Network
        #print(In.dtype)
        x = self.mel_network(In)
        x = self.emb_pool(self.embpoolpad(x))
        x = self.bnout(torch.flatten(x, start_dim=1))
        mel_out = self.emb_mel(x)
        
        #FFT-Network
        y = self.fft_network(In)
        fft_out = self.emb_fft(y)
        return torch.concat((fft_out,mel_out), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _wilkinghoff(melcfg, out_dims: int = 128, mdam: bool = True, **kwargs: Any) -> Wilkinghoff:
    model = Wilkinghoff(melcfg, out_dims, mdam, **kwargs)
    return model

