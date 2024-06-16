import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

#This is my Loss, it is not currently supported.
class WarpLoss(nn.Module):
    def __init__(self, cfg, n_classes=10, out_dims=256, learnable_loss=False, **kwargs):
        super(WarpLoss, self).__init__()
        self.num_classes = n_classes
        self.Warp = cfg['warp']
        self.k1 = cfg['k1'] 
        self.k2 = cfg['k2']
        self.alpha = cfg['alpha']
        self.Temp = cfg['temp']
        self.margin = cfg['margin']

        #Initialize Proxies
        Proxies = torch.randn(n_classes, out_dims)
        nn.init.kaiming_normal_(Proxies, mode='fan_out')
        self.Proxies = torch.nn.Parameter(Proxies, requires_grad=learnable_loss)
    
    def _UpdateParameters(self, cfg):
        self.k1 = cfg['k1'] 
        self.k2 = cfg['k2']
        self.alpha = cfg['alpha']
        self.Temp = cfg['temp']
    
    def _Warp(self, Distances, HotLabels, WarpMask):
        ColdLabels = 1.0 - HotLabels
        
        #Outward Warp
        NoMatchWarp = self.k1*HotLabels*Distances + ColdLabels*Distances
        Deltas = Distances - NoMatchWarp
        k1Warp = NoMatchWarp + self.margin*Deltas.detach()
        
        #Inward Warp
        #k2Warp = HotLabels*(self.k2*Distances + self.alpha*(1.0 - self.k2)) + ColdLabels*Distances
        NoMatchk2Warp = HotLabels*(self.k2*Distances + self.alpha*(1.0 - self.k2)) + ColdLabels*Distances
        k2Deltas = Distances - NoMatchk2Warp
        k2Warp = NoMatchk2Warp + k2Deltas.detach()
        #k2Warp = NoMatchk2Warp + (1.0 / self.margin)*k2Deltas.detach()

        #Combining the Inward and Outward Warps
        FinalWarp = self.Temp*WarpMask*k1Warp + self.Temp*(1.0 - WarpMask)*k2Warp
        return FinalWarp

    def _Distances(self, Output, Proxies):
        Proxies = Proxies.permute(1,0)
        #L2-Norm between Embeddings and Centroids
        HyperNorms = list()
        for i in range(Output.shape[0]):
            #HNorm = (Output[i].unsqueeze(-1) - Means[0])**2
            HNorm = (Output[i].unsqueeze(-1) - Proxies)**2
            HyperNorms.append(torch.sqrt(HNorm.sum(dim=0)))
        
        return torch.stack(HyperNorms, dim=0)

    #More optimized way to Calculate Euclidean Distances.
    def _euclidean_dist(self, x, y):
        """
        Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
        Returns:
        dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = dist - 2 * torch.matmul(x, y.t())
        # dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _compute_means(self, Embeddings, HotLabels):
        #Loop through object_num dimension and find centroid of every object (simultaneously for each sample in the batch)
        Means = list()
        Masks = list()
        Labels = torch.argmax(HotLabels, dim=-1)
        for ID in torch.unique(Labels):
            Mask = (Labels == ID).float()
            PairSpace = Embeddings*Mask.unsqueeze(-1)
            Mean = PairSpace.sum(dim=0) / (Mask.sum())# + self.epsilon)
            Means.append(Mean)
            Masks.append(Mask.long())
        #print(self.Proxies.shape)
        #print(Means[0].shape)
        Centroids = torch.stack(Means, dim=0)
        MaskLabels = torch.stack(Masks, dim=-1)
        #print(Centroids.shape)
        #input()
        return Centroids, MaskLabels
    
    def _embedding(self, Output, HotLabels):
        #L2-Distances between each Proxy
        """
        Means, RegLabels = self._compute_means(Output, HotLabels)
        RegNorms = self._euclidean_dist(Output, Means)
        RegLoss = torch.sum(RegLabels*RegNorms) / RegLabels.shape[0]
        """
        RegLoss = 0.0
        #"""
        
        #EuclideanNorms = self._Distances(Output, self.Proxies)
        EuclideanNorms = self._euclidean_dist(Output, self.Proxies)
        HotDistances = torch.max(HotLabels*EuclideanNorms, dim=1, keepdim=True)[0]# + 1.0

        if self.Warp:
            if True:#self.Sigmoid:
                loss_coeff = 100
                #import code
                #Softmax Warping (More numerically stable way of calculating loss)
                #WarpMask = (HotDistances < self.alpha).float().detach()
                #WarpedNorms = self._Warp(EuclideanNorms, HotLabels, WarpMask)
                #EuclideanNorms.register_hook(lambda grad: print("EuclideanNorms Gradient:",grad))
                #EuclideanNorms.register_hook(lambda grad: code.interact(local=locals()))
                WarpedNorms = 30*torch.sigmoid((EuclideanNorms - self.alpha) / 30)
                """
                R = torch.max(HotLabels*EuclideanNorms, dim=1)[0]
                print(R)
                print(R.mean())
                print(R.median())
                print(R.min())
                print(R.max())
                input()
                #"""
                #code.interact(local=locals())
                #WarpedNorms.register_hook(lambda grad: print("WarpedNorms Gradient:",grad))
                LogProbs = F.log_softmax(-self.Temp*WarpedNorms, dim=1)
                #LogProbs.register_hook(lambda grad: print("LogProbs Gradient:",grad))
                #import code
                #LogProbs.register_hook(lambda grad: code.interact(local=locals()))
            else:
                #Softmax Warping (More numerically stable way of calculating loss)
                loss_coeff = 1
                WarpMask = (HotDistances < self.alpha).float().detach()
                WarpedNorms = self._Warp(EuclideanNorms, HotLabels, WarpMask)
                LogProbs = F.log_softmax(-1*WarpedNorms, dim=1)

            CrossEntropy = torch.max(-1*HotLabels*LogProbs, dim=1)[0]
            return loss_coeff*(CrossEntropy.mean(dim=0) + 0.01*RegLoss)
        
        else:
            Default = HotDistances - EuclideanNorms
            Kernel = self.Temp*Default
        
        #Final Cross-Entropy Loss
        HotProbs = torch.sum(torch.exp(Kernel), dim=1)
        CrossEntropy = torch.log(HotProbs).mean(dim=0)
        return CrossEntropy + 0.00001*RegLoss
    
    def forward(self, Output, Labels):
        #Create batch-wise class masks off of Labels
        HotLabels = Labels#F.one_hot(Labels.long(), num_classes=self.num_classes)
        
        #Final-Loss
        FinalLoss = self._embedding(Output, HotLabels)

        return FinalLoss

class SCAdaCos(nn.Module):
    def __init__(self, n_classes=10, n_subclusters=1, out_dims=256, shift=8, learnable_loss=False, warp=True, **kwargs):
        super(SCAdaCos, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.out_dims = out_dims
        self.epsilon = 10**(-32)
        self.Warp = warp

        W = torch.randn((self.out_dims, self.n_classes*self.n_subclusters))
        self.W = torch.nn.Parameter(nn.init.xavier_uniform_(W), requires_grad=learnable_loss)
        self.s = torch.nn.Parameter(torch.Tensor([10]), requires_grad=False)
        #self.s = torch.nn.Parameter(torch.Tensor([math.sqrt(2) * math.log(n_classes*n_subclusters - 1)]), requires_grad=False)
        self.shift = torch.nn.Parameter(torch.Tensor([torch.pi / shift]), requires_grad=False)
        self.margin = 0.0
        
    
    def forward(self, x, y1):
        #print(self.s)
        #input("Scale Value")
        #y1 = F.one_hot(y1.long(), num_classes=self.n_classes)
        #x = output embeddings, y1 = mixed_up labels, y2 = original non-mixed_up labels #<--- This was the format before. I changed it. But I'm too scared to delete this.
        y1_orig = y1.clone()
        y1 = torch.repeat_interleave(y1, repeats=self.n_subclusters, dim=-1)
        # normalize feature
        #X = x.clone()
        x = F.normalize(x, dim=1)
        # normalize weights
        W = F.normalize(self.W, dim=0)
        # dot product
        logits = x @ W  # same as cos theta
        theta = torch.acos(torch.clip(logits, self.epsilon - 1.0, 1.0 - self.epsilon))
        #import code
        #code.interact(local=locals())
        """
        #Modified Adaptive Scale. Wilkinghoff's version exploded self.s when not using mixup. Changed "roughly" back to the way it was originally, Kind-of (somewhat). 
        B_avg = torch.exp(self.s*logits.detach())
        B_avg = torch.sum((y1 == 0).float()*B_avg, dim=1).mean(dim=0)
        
        theta_subclass = torch.min(theta.detach().view(-1, self.n_classes, self.n_subclusters), dim=2)[0]
        theta_class = torch.sum(y1_orig * theta_subclass, dim=1) # take mix-upped angle of mix-upped classes)
        theta_med = torch.median(theta_class) # computes median
        self.s[:] = torch.log(B_avg) / (torch.cos(torch.minimum((torch.pi / 4)*torch.ones_like(theta_med), theta_med)))# + self.epsilon)
        #print(self.s)
        #print(theta_med)
        #print(B_avg)
        #input()
        #"""
        #WT = W.unsqueeze(-1)
        #w_angles = torch.acos(torch.sum(W.unsqueeze(1)*WT, dim=0))
        #print(w_angles[0,:])
        #print((w_angles[0,:] / torch.pi)*180)
        #input()
        #HotLogits = torch.cos(theta - (torch.pi / 16))#/ 1.0)#3.0)# (torch.pi / 8))
        #Logits = 1.0*y1*HotLogits + (1.0 - y1)*logits
        #HotLogits = torch.cos(theta / 2.0)#3.0)# (torch.pi / 8))
        #Logits = 2.9*y1*HotLogits + (1.0 - y1)*logits
        def Cross_Entropy(CELogits):
            CELogits *= self.s
            sub_probs = F.softmax(CELogits - self.margin*y1_orig).view(-1, self.n_classes, self.n_subclusters)
            probs = torch.sum(sub_probs, dim=2)
            Hotprobs = torch.sum(y1_orig*probs, dim=1)
            Hotprobs[Hotprobs==0] += self.epsilon
            CE = -1*torch.log(Hotprobs)
            return CE.mean(dim=0)
            
        Shift = self.shift if self.Warp else 0.0
        Logits = torch.cos(theta - Shift)#(torch.pi / 4))
        #delta = logits - Logits
        #Logits = Logits + delta.detach()
        CE1 = Cross_Entropy(Logits)
        CE2 = Cross_Entropy(logits.clone().detach())
        #logits *= self.s
        #sub_probs = F.softmax(logits - self.margin*y1_orig).view(-1, self.n_classes, self.n_subclusters)
        #probs = torch.sum(sub_probs, dim=2)
        #Hotprobs = torch.sum(y1_orig*probs, dim=1)
        #Hotprobs[Hotprobs==0] += self.epsilon
        #CE = -1*torch.log(Hotprobs)

        return CE1, CE2#CE2 #CE.mean(dim=0)
        

if __name__=='__main__':
    print("Just a PyTorch Loss Function. Nothing to see here...")
