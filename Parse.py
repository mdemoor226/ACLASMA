import scipy.stats as st
import numpy as np


file = open("./testlog3.txt", "r")
Warp = False
Eval = False
NextLine = False
Seeds = []
AUCs = []
pAUCs = []
BestAUCs = []
BestpAUCs = []
WBestAUCs = []
WBestpAUCs = []
testAUCs = []
testpAUCs = []
WtestAUCs = []
WtestpAUCs = []
Line = file.readline()
while Line != '':
    #2024-07-02 20:32:23,637 Solace INFO: Applied Seed: 7218
    if Line[45:50] == "Seed:":
        seed = int(Line.split()[-1])
        if seed not in set(Seeds):
            print("Parsing seed {} logs...".format(seed))
            Seeds.append(seed)

    #Adaptive Scale: True || Warp: False || Scale Value: 8.000000 || Shift Value: 13.000000
    if Line[24:29] == "Warp:":
        Warp = eval(Line.split()[5])
        
    #2024-07-02 20:44:33,728 Solace INFO: mean AUC: 70.85239703857157
    #2024-07-02 20:44:33,729 Solace INFO: mean pAUC: 57.338001484864755
    #2024-07-02 20:44:33,729 Solace INFO: [71.94 61.13 72.08 57.26 70.85 57.34]
    if Line[37:46] == "mean AUC:":
        AUC = float(Line.split()[-1])
        if Eval:
            if Warp:
                WtestAUCs.append(AUC)
            else:
                testAUCs.append(AUC)
        else:
            AUCs.append(AUC)

    if Line[37:47] == "mean pAUC:":
        pAUC = float(Line.split()[-1])
        if Eval:
            if Warp:
                WtestpAUCs.append(pAUC)
            else:
                testpAUCs.append(pAUC)
        else:
            pAUCs.append(pAUC)
        
        NextLine = True
        Eval = False #Place Below Within "NextLine" Condition if it iis used. 

    if NextLine:
        #Add Later...(if desired).
        NextLine = False


    #2024-07-02 20:44:33,789 Solace INFO: Best Model Checkpoint: Epoch 8
    #print(Line[37:65])
    #input()
    if Line[37:65] == "Best Model Checkpoint: Epoch":
        #print(":)")
        A = np.array(AUCs)
        B = np.array(pAUCs)
        BestIndex = np.argmax((A + B) / 2) + 1
        #BestIndex = int(Line.split()[-1])
        if Warp:
            WBestAUCs.append(AUCs[BestIndex - 1])
            WBestpAUCs.append(pAUCs[BestIndex - 1])
        else:
            BestAUCs.append(AUCs[BestIndex - 1])
            BestpAUCs.append(pAUCs[BestIndex - 1])
        AUCs.clear()
        pAUCs.clear()
        Eval = True
    
    Line = file.readline()


file.close()


def get_stats(List):
    Array = np.array(List)
    mean = np.mean(Array)
    cf_low, cf_high = st.t.interval(0.95, len(Array)-1, loc=mean, scale=st.sem(Array)) #https://stackoverflow.com/a/34474255
    width = mean - cf_low
    return mean, width


print("################Development Set#######################")
#print("Mean non-Warped AUCs:\n",sum(BestAUCs) / len(BestAUCs))
Mean, Width = get_stats(BestAUCs) 
print("Non-Warped AUCs Confidence Interval:",round(Mean, 2),"+-",round(Width, 2))
#print("Mean Warped AUCs:",sum(WBestAUCs) / len(WBestAUCs))
Mean, Width = get_stats(WBestAUCs) 
print("Warped AUCs Confidence Interval:",round(Mean, 2),"+-",round(Width, 2))
#print("Mean non-Warped pAUCs:",sum(BestpAUCs) / len(BestpAUCs))
Mean, Width = get_stats(BestpAUCs) 
print("Non-Warped pAUCs Confidence Interval:",round(Mean, 2),"+-",round(Width, 2))
#print("Mean Warped pAUCs:",sum(WBestpAUCs) / len(WBestpAUCs))
Mean, Width = get_stats(WBestpAUCs) 
print("Warped pAUCs Confidence Interval:",round(Mean, 2),"+-",round(Width, 2))
print("#######################################################")

print("################Test/Evaluation Set####################")
#print("Mean non-Warped AUCs:",sum(testAUCs) / len(testAUCs))
Mean, Width = get_stats(testAUCs) 
print("Non-Warped AUCs Confidence Interval:",round(Mean, 2),"+-",round(Width, 2))
#print("Mean Warped AUCs:",sum(WtestAUCs) / len(WtestAUCs))
Mean, Width = get_stats(WtestAUCs) 
print("Warped AUCs Confidence Interval:",round(Mean, 2),"+-",round(Width, 2))
#print("Mean non-Warped pAUCs:",sum(testpAUCs) / len(testpAUCs))
Mean, Width = get_stats(testpAUCs) 
print("Non-Warped pAUCs Confidence Interval:",round(Mean, 2),"+-",round(Width, 2))
#print("Mean Warped pAUCs:",sum(WtestpAUCs) / len(WtestpAUCs))
Mean, Width = get_stats(WtestpAUCs) 
print("Warped pAUCs Confidence Interval:",round(Mean, 2),"+-",round(Width, 2))
print("#######################################################")


