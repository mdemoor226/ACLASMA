import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from model import Wilkinghoff
from loss import SCAdaCos, WarpLoss
from dataset.dataset import Data
from dataset.utils import BalancedFullBatchSampler
from dataset.utils import load_config, prepare_data
from Logger import SetupLogger
from eval import evaluate
import time
import datetime
#import random
import json
import argparse

parser = argparse.ArgumentParser(description='Project Solace')
parser.add_argument('--config', default='config.json')
parser.add_argument('--resume', default=None)
parser.add_argument('--seed', default=None)

args = parser.parse_args()
SEED = args.seed

if SEED is not None:
    seed = eval(args.seed)
    #For Debugging Purposes...
    #seed = 1717#random.randint(0,100000) 
    #print("Seed:",seed)
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

class Trainer(object):
    def __init__(self, cfg, ckpt_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert cfg['data_year'] in {2022, 2023}, "Error: data year must either be '2022' or '2023'. Entered: {}".format(cfg['data_year'])
        cfg['melcfg']['device'] = self.device
        self.epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        self.n_batch_classes = cfg['classes_per_batch']
        self.n_class_samples = self.batch_size // self.n_batch_classes
        self.n_subclusters = cfg['num_sub_clusters']

        #Prepare Dataset
        data_year = "./" + str(cfg['data_year']) + "-data/"
        TrData, EvData, UnkData, TsData, Meta = prepare_data(year=data_year, target_sr=cfg['target_sr'])
        print("Using {} (Non-SMOTE) training samples.".format(len(TrData['Data'])))
        print("Using {} validation samples.".format(EvData['Data'].shape[0]))
        
        #Train DataLoader
        SMOTE = cfg['smotecfg']['smote']
        self.Smote_refresh = cfg['smotecfg']['refresh']
        self.Smote_init_refresh = SMOTE and cfg['smotecfg']['init_refresh']
        self.Smote_rate = cfg['smotecfg']['refresh_rate']
        Train_Data = Data(TrData, Train=True, SMOTE=SMOTE, Target_sr=cfg['target_sr'], smotecfg=cfg['smotecfg'], year=cfg['data_year'])
        if cfg['class_balanced_sampling']:
            TrSampler = BalancedFullBatchSampler(TrData['Labels'], self.n_class_samples, self.batch_size)
            #self.Tr1Loader = data.DataLoader(Train_Data, batch_size=self.batch_size, shuffle=(TrSampler is None), batch_sampler=TrSampler, pin_memory=True) 
            self.Tr1Loader = data.DataLoader(Train_Data, batch_sampler=TrSampler, num_workers=2, pin_memory=True) 
            self.Tr2Loader = data.DataLoader(Train_Data, batch_sampler=TrSampler, num_workers=2, pin_memory=True)
        else:
            self.Tr1Loader = data.DataLoader(Train_Data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)#SMOTE) 
            self.Tr2Loader = data.DataLoader(Train_Data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)#SMOTE)

        self.Lambda = torch.distributions.beta.Beta(cfg['mixupcfg']['alpha'],cfg['mixupcfg']['alpha'])
        self.Mixup = cfg['mixupcfg']['mixup']
        
        #Train Set Eval DataLoader
        TrEval_Data = Data(TrData, Train=True, SMOTE=False, Target_sr=cfg['target_sr'], smotecfg=cfg['smotecfg'], year=cfg['data_year'])
        self.TrVLoader = data.DataLoader(TrEval_Data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False) 
        
        #Val Set Eval DataLoader
        Val_Data = Data(EvData, year=cfg['data_year'])
        self.ValLoader = data.DataLoader(Val_Data, batch_size=self.batch_size, num_workers=2, pin_memory=True, drop_last=False)
        
        #Other DataLoaders
        Unknown_Data = Data(UnkData, year=cfg['data_year'])
        Test_Data = Data(TsData, year=cfg['data_year'])
        self.UnknownLoader = data.DataLoader(Unknown_Data, batch_size=self.batch_size, num_workers=2, pin_memory=True, drop_last=False)
        self.TestLoader = data.DataLoader(Test_Data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

        #Combine all the Loaders into single object which is used for evaluation
        self.Meta = Meta
        self.Meta['TsFiles'] = TsData['Files']
        self.Meta['data_year'] = data_year
        self.Loaders = {'Train': self.TrVLoader, 'Eval': self.ValLoader, 'Unknown': self.UnknownLoader, 'Test': self.TestLoader}

        self.model = Wilkinghoff(melcfg=cfg['melcfg'], mdam=cfg['mdam'], data_year=data_year).to(self.device)
        if cfg['my_loss']:
            self.warpcfg = cfg['warpcfg']
            self.follow_cfg = cfg['warpcfg']['followup']
            self.Loss = WarpLoss(self.warpcfg, n_classes=Meta['num_trclasses'], out_dims=2*cfg['dimensions'], learnable_loss=cfg['learnable_loss']).to(self.device)
        else:
            self.Loss = SCAdaCos(n_classes=Meta['num_trclasses'], n_subclusters=self.n_subclusters, out_dims=2*cfg['dimensions'], scale=cfg['scale'], shift=cfg['shift_value'], 
                                 learnable_loss=cfg['learnable_loss'], adaptive_scale=cfg['adaptive_scale'], warp=cfg['warp']).to(self.device)

        #Load Checkpoint if given
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            #ProxyCheckpoint = "./Checkpoints/Loss-Weights/Best_weights.pth"
            ProxyCheckpoint = "./Checkpoints/Loss-Weights/Weights_Iter:" + ckpt_path[-8:] #Fix This Value if necessary
            proxy_checkpoint = torch.load(ProxyCheckpoint, map_location=self.device)
            self.Loss.load_state_dict(proxy_checkpoint)
    
        #Omit Batch Norm, bias, and non trainable parameters from weight decay.regularization
        Parameters = [{'params': [Param for Param in self.model.parameters() if len(Param.shape) == 1 and Param.requires_grad], 'weight_decay': 0.0, 'lr': cfg['learning_rate']},
                      {'params': [Param for Param in self.model.parameters() if len(Param.shape) != 1 and Param.requires_grad], 'weight_decay': cfg['weight_decay'], 'lr': cfg['learning_rate']},
                      {'params': [Param for Param in self.Loss.parameters()], 'weight_decay': cfg['loss_weight_decay'], 'lr': cfg['loss_learning_rate']}]
        #self.optimizer = torch.optim.Adam(Parameters)# if not cfg['my_loss'] else 
        self.optimizer = torch.optim.AdamW(Parameters)#, eps=1.0) #Change this...?
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        #Some job tracking variables
        self.epoch = 0
        self.iteration = 0
        self.epochs_per_eval = cfg['epochs_per_eval']
        self.iters_per_log = cfg['iterations_per_log']
        self.epochs_per_ckpt = cfg['epochs_per_ckpt']
        self.init_eval = cfg['init_eval']
        self.my_loss = cfg['my_loss']
        self.evalcfg = cfg['evalcfg']
        self.test_eval = cfg['test_eval']
        
        #Set the Logger so we know what's going on during the training process
        filename = "log.txt" if self.my_loss else "log3.txt"
        self.logger = SetupLogger(name="Solace", save_dir=".", distributed_rank=0, filename=filename, mode="a+")
        self.logger.info("Applied Seed: {}".format(SEED))
        if self.my_loss:
            self.logger.info(
                "Hyperparameters:\nalpha: {:f} || Temp: {:.4f} || k1: {:.4f} || k2: {:.4f} || lr: {} || plr: {} || f_alpha: {:f} || f_k1: {:.4} || f_2: {:.4} || f_temp: {:.4f} || flr: {} || fplr: {}".format(
                cfg['warpcfg']['alpha'], cfg['warpcfg']['temp'], cfg['warpcfg']['k1'], cfg['warpcfg']['k2'], cfg['learning_rate'], cfg['loss_learning_rate'],
                self.follow_cfg['alpha'], self.follow_cfg['k1'], self.follow_cfg['k2'], self.follow_cfg['temp'], self.follow_cfg['learning_rate'], self.follow_cfg['proxy_learning_rate'])
                )
        else:
            self.logger.info("Settings:\nAdaptive Scale: {} || Warp: {} || Scale Value: {:f} || Shift Value: {:f}".format(cfg['adaptive_scale'], cfg['warp'], 
                             cfg['scale'], cfg['shift_value']))
                    
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
        if False:#Test:
            #Load Best Checkpoitn
            #print("Evaluating Best Model on Test Data...")
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
                    if not self.my_loss:
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
        
        final_results = evaluate(embeddings, Preds, self.Meta, self.logger, self.evalcfg, n_subclusters=self.n_subclusters, Test=Test)
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
        
        if self.Smote_init_refresh:
            self.Tr1Loader.dataset.refresh(self.model)
            self.Tr2Loader.dataset.refresh(self.model)
        
        """
        self.Loss._UpdateParameters(self.warpcfg['followup'])
        #self.optimizer.param_groups[0]['lr'] = self.warpcfg['followup']['learning_rate']
        #self.optimizer.param_groups[1]['lr'] = self.warpcfg['followup']['learning_rate']
        #self.optimizer.param_groups[2]['lr'] = self.warpcfg['followup']['proxy_learning_rate']
        #"""

        #Best_Epoch = 1 #Temporarily removed for now, can add back in later.
        LossQueue = 100*[0.0]
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
                #print(Batch.shape)
                #print(torch.unique(torch.argmax(Labels, dim=-1)).shape)
                #input()
                #continue
                Output = self.model(Batch)
                Loss, LossDefault = self.Loss(Output, Labels)
                assert (Loss == Loss).all(), "Nans!!!"
                #Running Average of Loss
                LossQueue[-1] = LossDefault.item()#Loss.item()# / 100
                LossQueue = LossQueue[-1:] + LossQueue[:-1]
                running_avg = sum(LossQueue) / min(100,self.iteration)
                
                #Backward
                self.optimizer.zero_grad()
                Loss.backward()
                #print("Iteration:",self.iteration)
                #input()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                #torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
                self.optimizer.step()

                #Log some stuff
                if self.iteration % self.iters_per_log == 0:
                    synchronize()
                    timecheck2 = time.time()
                    timeleft = ((timecheck2 - timecheck1) / (i + 1))*(iters_per_epoch - i - 1)
                    eta = str(datetime.timedelta(seconds=int(timeleft)))
                    self.logger.info("Epoch: {:d}/{:d} eta: {} || Iteration: {:d} Lr: {:.6f} || LossLr: {:.4f} || Current Loss: {:.4f}".format(
                            epoch, self.epochs, eta,
                            self.iteration,
                            self.optimizer.param_groups[0]['lr'],
                            self.optimizer.param_groups[2]['lr'],
                            running_avg))
            
            #self.lr_scheduler.step()
            
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
                #input("Saving checkpoint...")
                self._checkpoint()
                #pass

            if self.Smote_refresh and epoch % self.Smote_rate == 0 and epoch != self.epochs:
                self.Tr1Loader.dataset.refresh(self.model)
                self.Tr2Loader.dataset.refresh(self.model)

            if self.my_loss:
                if epoch == self.epochs - self.warpcfg['followup']['epochs']:
                    self.Loss._UpdateParameters(self.warpcfg['followup'])
                    #self.optimizer.param_groups[0]['lr'] = self.warpcfg['followup']['learning_rate']
                    #self.optimizer.param_groups[1]['lr'] = self.warpcfg['followup']['learning_rate']
                    #self.optimizer.param_groups[2]['lr'] = self.warpcfg['followup']['proxy_learning_rate']
        
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
        self.logger.info("Trained for {:d} epochs. Goodbye.".format(self.epochs))
        input("Press ENTER to continue...")

def main(Args):
    #Initialize Trainer Class
    checkpoint_path = "./Checkpoints/" + Args.resume if Args.resume is not None else None
    TrManager = Trainer(load_config(Args.config), checkpoint_path)
    #random.seed(0)

    #Train the Model
    TrManager.train()
    
       
if __name__=='__main__':
    #Launch Training
    main(args)
