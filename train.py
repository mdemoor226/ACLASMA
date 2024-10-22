import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from model import Wilkinghoff
from loss import SCAdaCos
from dataset.dataset import Data
from dataset.utils import BalancedFullBatchSampler
from dataset.utils import load_config, prepare_data
from Logger import SetupLogger
from eval import evaluate
import time
import datetime
import json
import argparse

parser = argparse.ArgumentParser(description='Project Solace')
parser.add_argument('--config', default='config.json')
parser.add_argument('--resume', default=None)
parser.add_argument('--seed', default=None)
parser.add_argument('--warp', default=None)
parser.add_argument('--verbose', default=False)
parser.add_argument('--filename', default=None)

args = parser.parse_args()
SEED = args.seed
WARP = args.warp
VERBOSE = args.verbose
FILENAME = args.filename

if SEED is not None:
    seed = eval(args.seed)
    #For Debugging Purposes...
    #import random
    #seed = 1717#random.randint(0,100000) 
    #print("Seed:",seed)
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class Trainer(object):
    def __init__(self, cfg, ckpt_path=None):
        assert cfg['data_year'] in {2022, 2023}, "Error: data year must either be '2022' or '2023'. Entered: {}".format(cfg['data_year'])
        assert all(isinstance(Epoch, int) for Epoch in cfg['shiftcfg']['decay_epochs']), "Error: Decay Epoch values must be Integers."
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if WARP is not None:
            cfg['warp'] = eval(WARP)
        cfg['melcfg']['device'] = self.device
        self.epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        self.n_batch_classes = cfg['classes_per_batch']
        self.n_class_samples = self.batch_size // self.n_batch_classes
        self.n_subclusters = cfg['num_sub_clusters']

        #Prepare Dataset
        data_year = "./" + str(cfg['data_year']) + "-data/"
        TrData, EvData, UnkData, TsData, Meta = prepare_data(year=data_year)
        print("Using {} training samples.".format(len(TrData['Data'])))
        print("Using {} validation samples.".format(EvData['Data'].shape[0]))
        
        #Train DataLoader
        Train_Data = Data(TrData, Train=True)
        if cfg['class_balanced_sampling']:
            TrSampler1 = BalancedFullBatchSampler(TrData['Labels'], self.n_class_samples, self.batch_size)
            TrSampler2 = BalancedFullBatchSampler(TrData['Labels'], self.n_class_samples, self.batch_size)
            self.Tr1Loader = data.DataLoader(Train_Data, batch_sampler=TrSampler1, num_workers=2, pin_memory=True) 
            self.Tr2Loader = data.DataLoader(Train_Data, batch_sampler=TrSampler2, num_workers=2, pin_memory=True)
        else:
            self.Tr1Loader = data.DataLoader(Train_Data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
            self.Tr2Loader = data.DataLoader(Train_Data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        self.Mixup = cfg['mixupcfg']['mixup']
        self.Lambda = torch.distributions.beta.Beta(cfg['mixupcfg']['alpha'],cfg['mixupcfg']['alpha']) if self.Mixup else None
        
        #Train Set Eval DataLoader
        TrEval_Data = Data(TrData, Train=True)
        self.TrVLoader = data.DataLoader(TrEval_Data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False) 
        
        #Val Set Eval DataLoader
        Val_Data = Data(EvData)
        self.ValLoader = data.DataLoader(Val_Data, batch_size=self.batch_size, num_workers=2, pin_memory=True, drop_last=False)
        
        #Other DataLoaders
        Unknown_Data = Data(UnkData)
        Test_Data = Data(TsData)
        self.UnknownLoader = data.DataLoader(Unknown_Data, batch_size=self.batch_size, num_workers=2, pin_memory=True, drop_last=False)
        self.TestLoader = data.DataLoader(Test_Data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

        #Combine all the Loaders into single object which is used for evaluation
        self.Meta = Meta
        self.Meta['verbose'] = VERBOSE
        self.Meta['TsFiles'] = TsData['Files']
        self.Meta['data_year'] = data_year
        self.Loaders = {'Train': self.TrVLoader, 'Eval': self.ValLoader, 'Unknown': self.UnknownLoader, 'Test': self.TestLoader}
        
        self.DECAY = cfg['shiftcfg']['shift_decay']
        self.SFACTOR = cfg['shiftcfg']['decay_factor']
        self.SEPOCHS = set(cfg['shiftcfg']['decay_epochs'])
        self.model = Wilkinghoff(melcfg=cfg['melcfg'], mdam=cfg['mdam'], data_year=data_year).to(self.device)
        self.Loss = SCAdaCos(n_classes=Meta['num_trclasses'], n_subclusters=self.n_subclusters, out_dims=2*cfg['dimensions'], scale=cfg['scale'], shift=cfg['shiftcfg']['shift_value'], 
                             learnable_loss=cfg['learnable_loss'], adaptive_scale=cfg['adaptive_scale'], warp=cfg['warp']).to(self.device)

        #Load Checkpoint if given
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            #ProxyCheckpoint = "./Checkpoints/Loss-Weights/Best_weights.pth"
            ProxyCheckpoint = "./Checkpoints/Loss-Weights/Weights_Iter:" + ckpt_path[-8:] #Fix This Value if necessary
            proxy_checkpoint = torch.load(ProxyCheckpoint, map_location=self.device)
            self.Loss.load_state_dict(proxy_checkpoint)
        
        #Load in Pre-Trained Proxies/Class Centers if specified.
        if cfg['pre_trained_loss']:
            assert os.path.isfile("./Checkpoints/Loss-Weights/StartWeights.pth"), "Error, Training Runs that use pre-trained Proxies need pre-trained Weights. Make some and move the trained file to ./Checkpoints/Loss-Weights/StartWeights.pth."
            NewShift = self.Loss.shift.item()
            NewScale = self.Loss.s.clone()
            ProxyCheckpoint = "./Checkpoints/Loss-Weights/StartWeights.pth"
            proxy_checkpoint = torch.load(ProxyCheckpoint, map_location=self.device)
            self.Loss.load_state_dict(proxy_checkpoint)
            self.Loss.shift.data = torch.Tensor([NewShift])
            self.Loss.s.data = NewScale.data
            self.Loss.to(self.device)
        
        #Omit Batch Norm, bias, and non trainable parameters from weight decay.regularization
        Parameters = [{'params': [Param for Param in self.model.parameters() if len(Param.shape) == 1 and Param.requires_grad], 'weight_decay': 0.0, 'lr': cfg['learning_rate']},
                      {'params': [Param for Param in self.model.parameters() if len(Param.shape) != 1 and Param.requires_grad], 'weight_decay': cfg['weight_decay'], 'lr': cfg['learning_rate']},
                      {'params': [Param for Param in self.Loss.parameters()], 'weight_decay': cfg['loss_weight_decay'], 'lr': cfg['loss_learning_rate']}]
        self.optimizer = torch.optim.AdamW(Parameters)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0.0005)
        
        #Some job tracking variables
        self.epoch = 0
        self.iteration = 0
        self.epochs_per_eval = cfg['epochs_per_eval']
        self.iters_per_log = cfg['iterations_per_log']
        self.epochs_per_ckpt = cfg['epochs_per_ckpt']
        self.init_eval = cfg['init_eval']
        self.test_eval = cfg['test_eval']
        
        #Set the Logger so we know what's going on during the training process
        filename = FILENAME if FILENAME is not None else "log.txt"
        self.logger = SetupLogger(name="Solace", save_dir=".", distributed_rank=0, filename=filename, mode="a+")
        self.logger.info("Applied Seed: {}".format(SEED))
        self.logger.info("Settings:\nAdaptive Scale: {} || Warp: {} || Scale Value: {:f} || Shift Value: {:f}".format(cfg['adaptive_scale'], cfg['warp'], 
                        cfg['scale'], cfg['shiftcfg']['shift_value']))
                    
    def _checkpoint(self, model_name=None, proxy_name=None):
        #Save Checkpoint
        Model = self.model
        Dir = os.path.expanduser("./Checkpoints/")
        if not os.path.exists(Dir):
            os.makedirs(Dir)
        
        file = "Solace_Iter:{}.pth".format(self.iteration) if model_name is None else model_name
        CHECKPOINT_PATH = os.path.join(Dir,file)
        torch.save(Model.state_dict(), CHECKPOINT_PATH)
    
        Loss = self.Loss
        LDir = os.path.expanduser("./Checkpoints/Loss-Weights/")
        if not os.path.exists(LDir):
            os.makedirs(LDir)
        
        file = "Weights_Iter:{}.pth".format(self.iteration) if proxy_name is None else proxy_name
        ProxyCHECKPOINT_PATH = os.path.join(LDir,file)
        torch.save(Loss.state_dict(), ProxyCHECKPOINT_PATH)

    def _eval(self, Test=False):
        if Test:
            #Load Best Checkpoitn
            self.logger.info("Evaluating Best Model on Test Data...")
            checkpoint = torch.load("./Checkpoints/Best_model.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint)
            ProxyCheckpoint = "./Checkpoints/Loss-Weights/Best_weights.pth" #Fix This Value if necessary
            proxy_checkpoint = torch.load(ProxyCheckpoint, map_location=self.device)
            self.Loss.load_state_dict(proxy_checkpoint)
        
        self.model.eval()
        print("Evaluating Model...")
        
        #Begin Evaluation
        embeddings = {key : None for key in self.Loaders.keys()}
        with torch.no_grad():
            for Key, Loader in self.Loaders.items():
                Eval = []
                print("Obtaining {} Data Predictions...".format(Key))
                for i,Sample in enumerate(Loader):
                    #Send Data to GPU
                    Batch = Sample['Data'].to(self.device)
                    Labels = torch.nn.functional.one_hot(Sample['Labels'][:,0].to(self.device).long(), num_classes=self.Meta['num_trclasses'])
                    
                    #Forward through model
                    Output = self.model(Batch)
                    #Loss = self.Loss(Output, Labels)
                    Output = torch.nn.functional.normalize(Output, dim=1)
                    
                    #Gonvert Annotations to numpy arrays for post-processing
                    #Eval.append(torch.nn.functional.normalize(Output, dim=1).cpu().numpy())
                    Eval.append(Output.cpu().numpy())
                
                #Post-Process
                embeddings[Key] = np.concatenate(Eval, axis=0)
        
        Preds = defaultdict(list)
        Trshape = np.unique(self.Meta['Labels']['Train']).shape[0]
        Preds['Eval'] = np.zeros((self.Meta['Shapes']['Eval'][0], Trshape)) 
        Preds['Unknown'] = np.zeros((self.Meta['Shapes']['Unknown'][0], Trshape))
        Preds['Test'] = np.zeros((self.Meta['Shapes']['Test'][0], Trshape))
        Preds['Train'] = np.zeros((self.Meta['Labels']['Train'].shape[0], Trshape))
        
        final_results = evaluate(embeddings, Preds, self.Meta, self.logger, n_subclusters=self.n_subclusters, Test=Test)
        self.model.train()  
        return final_results
    
    def train(self):
        #Log the beginning of training
        self.logger.info("Training started for {:d} epochs.".format(self.epochs))
        self.model.train()
        
        #Pre-Evaluation
        Best_score, Best_epoch = 0.0, 0
        if self.init_eval:
            final_results_dev, _ = self._eval()
            _, final_results_eval = self._eval(Test=True)
            Best_score = np.mean(final_results_dev)
        
        #Best_Epoch = 1 #Temporarily removed for now, can add back in later.
        LossQueue = 100*[0.0]
        LossQueue2 = 100*[0.0]
        iters_per_epoch = len(self.Tr1Loader)
        #iters_per_epoch = int((self.batch_size/(self.n_batch_classes*self.n_class_samples))*len(self.Tr1Loader))
        synchronize = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        for epoch in range(1,self.epochs+1):
            self.epoch = epoch
            running_avg = 0
            
            synchronize()
            timecheck1 = time.time()
            for i,Samples in enumerate(zip(self.Tr1Loader,self.Tr2Loader)):
                self.iteration += 1

                #Optionally apply mixup. Send Sample to GPU.
                if self.Mixup:
                    Batch1 = Samples[0]['Data'].to(self.device)
                    Batch2 = Samples[1]['Data'].to(self.device)
                    Labels1 = torch.nn.functional.one_hot(Samples[0]['Labels'][:,0].to(self.device).long(), num_classes=self.Meta['num_trclasses'])
                    Labels2 = torch.nn.functional.one_hot(Samples[1]['Labels'][:,0].to(self.device).long(), num_classes=self.Meta['num_trclasses'])
                    Lambda = self.Lambda.sample().to(self.device)
                    Batch = Lambda*Batch1 + (1.0 - Lambda)*Batch2
                    Labels = Lambda*Labels1 + (1.0 - Lambda)*Labels2
                else:
                    #Batch = Samples['Data'].to(self.device)
                    #Labels = torch.nn.functional.one_hot(Samples['Labels'][:,0].to(self.device).long(), num_classes=self.Meta['num_trclasses'])
                    Batch = Samples[0]['Data'].to(self.device)
                    Labels = torch.nn.functional.one_hot(Samples[0]['Labels'][:,0].to(self.device).long(), num_classes=self.Meta['num_trclasses'])
                
                #Forward
                Output = self.model(Batch)
                Loss, LossDefault = self.Loss(Output, Labels)
                assert (Loss == Loss).all(), "Nans!!!"
                
                #Running Average of Loss
                LossQueue[-1] = LossDefault.item()
                LossQueue2[-1] = Loss.item()
                LossQueue = LossQueue[-1:] + LossQueue[:-1]
                LossQueue2 = LossQueue2[-1:] + LossQueue2[:-1]
                running_avg = sum(LossQueue) / min(100,self.iteration)
                running_avg2 = sum(LossQueue2) / min(100,self.iteration)
                
                #Backward
                self.optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                #torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
                self.optimizer.step()

                #Log some stuff
                if self.iteration % self.iters_per_log == 0:
                    synchronize()
                    timecheck2 = time.time()
                    timeleft = ((timecheck2 - timecheck1) / (i + 1))*(iters_per_epoch - i - 1)
                    eta = str(datetime.timedelta(seconds=int(timeleft)))
                    self.logger.info("Epoch: {:d}/{:d} eta: {} || Iteration: {:d} Lr: {:.6f} || LossLr: {:.4f} || Default Loss: {:.4f} || Warp Loss: {:.4f}".format(
                            epoch, self.epochs, eta,
                            self.iteration,
                            self.optimizer.param_groups[0]['lr'],
                            self.optimizer.param_groups[2]['lr'],
                            running_avg,
                            running_avg2))
            
            self.lr_scheduler.step()
            if self.DECAY and self.epoch in self.SEPOCHS:
                print("Adjusting Shift Parameter...")
                self.Loss.shift *= self.SFACTOR
            self.logger.info("Current Shift Parameter {:.4f}".format((180*(self.Loss.shift / torch.pi)).item()))
            self.logger.info("Current Loss Scale Term: {}".format(self.Loss.s.item()))
            
            #Run evalution if enough epochs have passed
            if epoch % self.epochs_per_eval == 0:
                final_results_dev, final_results_eval = self._eval()
                
                #Add lines here to log the result/keep track of the best results so far...
                Score = np.mean(final_results_dev)
                if Score > Best_score:
                    Best_score = Score
                    Best_epoch = epoch
                
                    self._checkpoint(model_name="Best_model.pth", proxy_name="Best_weights.pth")
                
            if epoch % self.epochs_per_ckpt == 0:
                #print("Iteration:",self.iteration)
                #print("Saving checkpoint...")
                self._checkpoint()
        
        #Save the Checkpoint
        self._checkpoint()
        
        #Final Evaluation...
        if self.epochs % self.epochs_per_eval > 0:
            self._eval()
        
        self.logger.info("#########################################################")
        self.logger.info("########### Finished Training and Evaluation ############")
        self.logger.info("#########################################################")
        self.logger.info("Best Model Checkpoint: Epoch {:d}".format(Best_epoch))
        if self.test_eval:
            final_results_dev, final_results_eval = self._eval(Test=True)
        
        #Training is finished
        self.logger.info("Trained for {:d} epochs. Goodbye.\n".format(self.epochs))
        #input("Press ENTER to continue...")

def main(Args):
    #Initialize Trainer Class
    checkpoint_path = "./Checkpoints/" + Args.resume if Args.resume is not None else None
    TrManager = Trainer(load_config(Args.config), checkpoint_path)

    #Train the Model
    TrManager.train()
    
       
if __name__=='__main__':
    #Launch Training
    main(args)
