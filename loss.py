import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

class SCAdaCos(nn.Module):
    def __init__(self, n_classes=10, n_subclusters=1, out_dims=256, scale=10, shift=8, learnable_loss=False, adaptive_scale=False, warp=True, **kwargs):
        super(SCAdaCos, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.out_dims = out_dims
        self.epsilon = 10**(-32)
        self.Warp = warp
        self.Adaptive_s = adaptive_scale

        W = torch.randn((self.out_dims, self.n_classes*self.n_subclusters))
        self.W = torch.nn.Parameter(nn.init.xavier_uniform_(W), requires_grad=learnable_loss)
        if self.Adaptive_s:
            self.s = torch.nn.Parameter(torch.Tensor([math.sqrt(2) * math.log(self.n_classes*self.n_subclusters - 1)]), requires_grad=False)
        else:
            self.s = torch.nn.Parameter(torch.Tensor([scale]), requires_grad=False)
        self.shift = torch.nn.Parameter(torch.Tensor([torch.pi / shift]), requires_grad=False)
    
    def forward(self, x, y1):
        y1_orig = y1.clone()
        y1 = torch.repeat_interleave(y1, repeats=self.n_subclusters, dim=-1)
        # normalize feature
        x = F.normalize(x, dim=1)
        # normalize weights
        W = F.normalize(self.W, dim=0)
        # dot product
        logits = x @ W  # same as cos theta
        theta = torch.acos(torch.clip(logits, self.epsilon - 1.0, 1.0 - self.epsilon))
        if self.Adaptive_s:
            #Modified Adaptive Scale. Wilkinghoff's version exploded self.s when not using mixup. Changed "roughly" back to the way it was originally
            B_avg = torch.exp(self.s*logits.detach())
            B_avg = torch.sum((y1 == 0).float()*B_avg, dim=1).mean(dim=0)
            
            theta_subclass = torch.min(theta.detach().view(-1, self.n_classes, self.n_subclusters), dim=2)[0]
            theta_class = torch.sum(y1_orig * theta_subclass, dim=1) 
            theta_med = torch.median(theta_class) # computes median
            self.s[:] = torch.log(B_avg) / (torch.cos(torch.minimum((torch.pi / 4)*torch.ones_like(theta_med), theta_med)))# + self.epsilon)
        
        def Cross_Entropy(CELogits):
            CELogits *= self.s
            sub_probs = F.softmax(CELogits).view(-1, self.n_classes, self.n_subclusters)
            probs = torch.sum(sub_probs, dim=2)
            Hotprobs = torch.sum(y1_orig*probs, dim=1)
            Hotprobs[Hotprobs==0] += self.epsilon
            CE = -1*torch.log(Hotprobs)
            return CE.mean(dim=0)
            
        Shift = self.shift if self.Warp else 0.0
        Logits = torch.cos(theta - Shift)
        CE1 = 10*Cross_Entropy(Logits)
        CE2 = 10*Cross_Entropy(logits.clone().detach()) #Keep Track of (Default) Non-Shifted Loss Function. 
        
        return CE1, CE2
        
if __name__=='__main__':
    print("Just a PyTorch Loss Function. Nothing to see here...")
