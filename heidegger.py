import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats
import re
import time
import subprocess
import os
from features import *
import scipy.signal as ss
import numpy as np
import math
from numpy import diff
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.extmath import cartesian
import json
import logging
import os
from functools import partial
from multiprocessing.pool import Pool
from time import time
import multiprocessing
from numpy.random import choice
from numpy import genfromtxt
l = multiprocessing.Lock()


dfPath = "df.pkl"
nanLabelPath= "nanLabel.pkl"

trtmntVar = set(["scfrda","scfrdg","scfrdm","heacta", "heactb","heactc", "scorg03","scorg06","scorg05","scorg07"])
confoundersVar = set(["indager", "hehelf","dhsex","totwq10_bu_s"])
binaryVariables = set(["scorg03","scorg06","scorg05","scorg07","dhsex"])
targetVar = set(["memIndex"]) 
auxVar = set(["cfdscr","cflisen", "cflisd","cfdatd"])
drvVar = set(["memIndexChange", "baseMemIndex"])
allVar = trtmntVar|confoundersVar|targetVar
WAVE_NUM=7
reverseVars = ["heacta", "heactb", "heactc", "scfrdg", "scfrda", "hehelf"]


basePath = "/home/ali/Downloads/UKDA-5050-stata (2)/stata/stata13_se"
REFUSAL=-9
DONT_KNOW=-8
NOT_APPLICABLE=-1
SCHD_NOT_APPLICABLE=-2

NOT_ASKED=-3

NOT_IMPUTED = -999.0
NON_SAMPLE = -998.0
INST_RESPONDENT=  -995.0


def harmonizeData(df):
    # print allVar
    for var in (trtmntVar|confoundersVar):
        df[var] = df[var].apply((globals()[var].harmonize))

    return df


def binarizeData(df):
    # pattern = r"[a-zA-Z0-9_]*_n$"
    # for var in trtmntVar:
    #   if not re.match(pattern, var):  
    #       col_bin = var + '_b'
    #       df[col_bin] = df[var].apply((globals()[var].binarize))
    return df


def normalizeData(df):
    for var in (trtmntVar|confoundersVar|targetVar):
        dfs=[]
        for i in range(1,8):
            col = "{}_{}".format(var,i)
            dfs.append(pd.DataFrame( {var: df[col]}))
        mergedDf = pd.concat(dfs)
        nonZeros = mergedDf[mergedDf[var]!=0][var]
        NZMean = nonZeros.mean()
        mean= mergedDf[var].mean()
        std = mergedDf[var].std()
        minValue = mergedDf[var].min()
        maxValue = mergedDf[var].max()
        mergedDf.to_csv("rawValues_{}".format(var), index=False)
        for i in range(1,8):
            col = "{}_{}".format(var,i)
            col_norm = "{}_n_{}".format(var,i)
            if var == "scfrdm":
                df[col_norm] = 1/(1+ np.exp(-(df[col]-(NZMean/2))*(3))) 
            else:
                df[col_norm] = (df[col] - minValue)/(maxValue - minValue)
        if not var in targetVar:
            for i in range(1,8):
                col = "{}_{}".format(var,i)
                df = df.drop(columns=col)
    return df


def readWave1Data(basePath):
    waveNumber=1
    Core = pd.read_stata("{}/wave_1_core_data_v3.dta".format(basePath, waveNumber),convert_categoricals=False)
    Drv =  pd.read_stata("{}/wave_1_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
    FinDrv = pd.read_stata('{}/wave_1_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
    
    s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
    df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])
    
    df = df.rename(columns = {'scorg3':'scorg03'})
    df = df.rename(columns = {'scorg5':'scorg05'})
    df = df.rename(columns = {'scorg6':'scorg06'})
    df = df.rename(columns = {'scorg7':'scorg07'})
    
    col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
    df = df [col_list] 

    df = addMemIndex(df)
    df = removeAuxVars(df)

    df = harmonizeData(df)
    df = binarizeData(df)
    return df

def addSuffix(df, num):
    for var in (trtmntVar|confoundersVar|targetVar):
        newName = "{}_{}".format(var,num)
        df = df.rename(columns = {var:newName})
    return df


def readWave2Data(basePath):
    waveNumber=2
    Core = pd.read_stata("{}/wave_2_core_data_v4.dta".format(basePath, waveNumber),convert_categoricals=False)
    Drv =  pd.read_stata("{}/wave_2_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
    FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)

    s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
    df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])

    df = df.rename(columns = {'HeActa':'heacta'})
    df = df.rename(columns = {'HeActb':'heactb'})
    df = df.rename(columns = {'HeActc':'heactc'})
    df = df.rename(columns = {'Hehelf':'hehelf'})
    df = df.rename(columns = {'DhSex':'dhsex'})

    df = df.rename(columns = {'CfDScr':'cfdscr'})
    df = df.rename(columns = {'CfLisEn':'cflisen'})
    df = df.rename(columns = {'CfLisD':'cflisd'})
    df = df.rename(columns = {'CfDatD':'cfdatd'})

    col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
    df = df [col_list] 

    df = addMemIndex(df)
    df = removeAuxVars(df)

    df = harmonizeData(df)
    df = binarizeData(df)
    return df


def readWave3Data(basePath):
    waveNumber=3
    Core = pd.read_stata("{}/wave_{}_elsa_data.dta".format(basePath, waveNumber),convert_categoricals=False)
    Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
    FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
    
    s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
    df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])

    df = df.rename(columns = {'hegenh':'hehelf'})
    col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
    df = df [col_list] 

    df = addMemIndex(df)
    df = removeAuxVars(df)

    df = harmonizeData(df)
    df = binarizeData(df)
    return df


def readWave4Data(basePath):
    waveNumber=4
    Core = pd.read_stata("{}/wave_{}_elsa_data.dta".format(basePath, waveNumber),convert_categoricals=False)
    Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
    FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
    
    s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
    df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])

    col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
    df = df [col_list] 

    df = addMemIndex(df)
    df = removeAuxVars(df)

    df = harmonizeData(df)
    df = binarizeData(df)
    return df


def readWave5Data(basePath):
    waveNumber=5
    Core = pd.read_stata("{}/wave_{}_elsa_data.dta".format(basePath, waveNumber),convert_categoricals=False)
    Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
    FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)

    s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
    df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])

    col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
    df = df [col_list] 

    df = addMemIndex(df)
    df = removeAuxVars(df)

    df = harmonizeData(df)
    df = binarizeData(df)
    return df


def readWave6Data(basePath):
    waveNumber=6
    w6Core = pd.read_stata("{}/wave_{}_elsa_data_v2.dta".format(basePath, waveNumber),convert_categoricals=False)
    w6Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
    w6FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
    
    s1 = pd.merge(w6Core, w6Drv, how='inner', on=['idauniq'])
    df = pd.merge(s1, w6FinDrv, how='inner', on=['idauniq'])

    df = df.rename(columns = {'HeActa':'heacta'})
    df = df.rename(columns = {'HeActb':'heactb'})
    df = df.rename(columns = {'HeActc':'heactc'})
    df = df.rename(columns = {'Hehelf':'hehelf'})
    df = df.rename(columns = {'DhSex':'dhsex'})

    df = df.rename(columns = {'CfDScr':'cfdscr'})
    df = df.rename(columns = {'CfLisEn':'cflisen'})
    df = df.rename(columns = {'CfLisD':'cflisd'})
    df = df.rename(columns = {'CfDatD':'cfdatd'})

    col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
    df = df [col_list] 

    df = addMemIndex(df)
    df = removeAuxVars(df)

    df = harmonizeData(df)
    df = binarizeData(df)
    return df


def readWave7Data(basePath):
    waveNumber=7
    w6Core = pd.read_stata("{}/wave_{}_elsa_data.dta".format(basePath, waveNumber),convert_categoricals=False)
    w6Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
    w6FinDrv = pd.read_stata('{}/wave_7_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
    
    s1 = pd.merge(w6Core, w6Drv, how='inner', on=['idauniq'])
    df = pd.merge(s1, w6FinDrv, how='inner', on=['idauniq'])

    df = df.rename(columns = {'HeActa':'heacta'})
    df = df.rename(columns = {'HeActb':'heactb'})
    df = df.rename(columns = {'HeActc':'heactc'})
    df = df.rename(columns = {'Hehelf':'hehelf'})
    df = df.rename(columns = {'DhSex':'dhsex'})
    df = df.rename(columns = {'scfrdl':'scfrdm'})

    df = df.rename(columns = {'CfDScr':'cfdscr'})
    df = df.rename(columns = {'CfLisEn':'cflisen'})
    df = df.rename(columns = {'CfLisD':'cflisd'})
    df = df.rename(columns = {'CfDatD':'cfdatd'})
    
    col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
    df = df [col_list] 

    df = addMemIndex(df)
    df = removeAuxVars(df)

    df = harmonizeData(df)
    df = binarizeData(df)
    return df


def removeAuxVars(df):
    df= df.drop(list(auxVar),axis=1)
    return df


def readData(mergeMethod="outer"):
    df1 = readWave1Data(basePath)
    df2 = readWave2Data(basePath)
    df3 = readWave3Data(basePath)
    df4 = readWave4Data(basePath)
    df5 = readWave5Data(basePath)
    df6 = readWave6Data(basePath)
    df7 = readWave7Data(basePath)

    df12 = pd.merge(df1,  df2, how=mergeMethod, on=['idauniq'],suffixes=('_1', ''))
    df13 = pd.merge(df12, df3, how=mergeMethod, on=['idauniq'],suffixes=('_2', ''))
    df14 = pd.merge(df13, df4, how=mergeMethod, on=['idauniq'],suffixes=('_3', ''))
    df15 = pd.merge(df14, df5, how=mergeMethod, on=['idauniq'],suffixes=('_4', ''))
    df16 = pd.merge(df15, df6, how=mergeMethod, on=['idauniq'],suffixes=('_5', ''))
    df17 = pd.merge(df16, df7, how=mergeMethod, on=['idauniq'],suffixes=('_6', '_7'))

    return df17


def addMemIndex(df):
    df["memIndex"] = df.apply(computeMemIndex, axis=1)
    df= df.dropna(subset=["memIndex"])
    return df


def computeMemIndexChange(row, waveNumber):
    memtotVarCur = "memIndex_{}".format(waveNumber) 
    memtotVarPrev = "memIndex_{}".format(waveNumber-1)
    return row[memtotVarCur] - row[memtotVarPrev]


def computeMemIndex(row):
    if row["cfdatd"] == REFUSAL:
        return np.nan
    if row ["cflisd"] == DONT_KNOW:
        row ["cflisd"] = 0
    if row ["cflisen"] == DONT_KNOW:
        row ["cflisen"] = 0
    if (row ["cfdscr"]<0) or (row ["cflisd"]<0) or (row ["cflisen"]<0):
        return np.nan
    else:
        return row["cfdscr"] + row["cflisd"] + row["cflisen"]


def doRandomMatching(k, labels, trtNUM):
    indices = np.where(labels==k)[0]
    trt = indices[np.where(indices<trtNUM)[0]]
    ctrl = indices[np.where(indices>=trtNUM)[0]]
    if (len(trt)<len(ctrl)):
        ctrl = np.random.choice(ctrl, len(trt), replace=False)
    else:
        trt = np.random.choice(trt, len(ctrl), replace=False)

    res= zip(trt,ctrl)
    pairs = [p for p in res]    
    return pairs
    

def performMatching_RBD(C, trtNUM):
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, linkage='complete', distance_threshold=0.1).fit(C)
    labels = model.labels_
    CLUSTER_NUM = len(np.unique(labels))
    finalPairs = []
    for k in range(0, CLUSTER_NUM):
        pairs= doRandomMatching(k, labels, trtNUM)
        finalPairs = finalPairs+ pairs
    costs = []
    for pair in finalPairs:
        costs.append(C[pair[0], pair[1]])
    costs = np.array(costs)
    return finalPairs


def computePValue(X,Y):
    if np.array_equal(X,Y):
        return 2
    res= scipy.stats.wilcoxon(X,Y,"wilcox")
    pVal = res[1]
    return pVal


def expandData(df):
    newDF =  pd.DataFrame( {'idauniq':df["idauniq"]})
    for var in (trtmntVar|confoundersVar|targetVar):
        for i in range(1,8):
            col = "{}_n_{}".format(var,i)
            newDF[col]=np.nan
    for i in range(1,8):
        for var in targetVar:
            col = "{}_{}".format(var,i)
            newDF[col]= np.nan
    offset=7
    for var in (trtmntVar|confoundersVar|targetVar):
        for i in range(1,8):
            col = "{}_n_{}".format(var,i)
            newCol = "{}_n_{}".format(var,i+offset)
            df = df.rename(columns={col: newCol})

    for i in range(1,8):
        for var in targetVar:
            col = "{}_{}".format(var,i)
            newCol = "{}_{}".format(var,i+offset)       
            df = df.rename(columns={col: newCol})

    finalDF = pd.merge(newDF,  df, how="inner", on=['idauniq'])     
    return finalDF


def interpolate(df):
    for index, row in df.iterrows():
        for var in (trtmntVar|confoundersVar|targetVar):
            cols= []
            for i in range(1,15):
                col = "{}_n_{}".format(var,i)
                cols.append(col)
            seq = df.loc[index, cols]
            f2  = pd.DataFrame(np.array([seq]), columns=cols)
            f2  = f2.interpolate(method ='linear', axis=1, limit_direction="both" )
            df.loc[index, cols] = f2.loc[0, cols]
    return df


def computeDistance(seq1, seq2, seq1Label, seq2Label=None, weights=None):
    alpha = 0.3
    weights = weights[-len(seq1):]
    penalty =0.5
    sumDiff=0
    for i in range(len(seq1)):
        if seq1Label[i]:
            diff =np.abs(seq1[i]- seq2[i])
            diff =  (1-2*alpha)*diff+alpha
        elif (not seq2Label is None) and seq2Label[i]:
            diff =np.abs(seq1[i]- seq2[i])
            diff =  (1-2*alpha)*diff+alpha
        else:
            diff =np.abs(seq1[i]- seq2[i])
        if np.isnan(diff):
            diff= 1
        sumDiff = sumDiff + diff*weights[i]
    avgDiff = sumDiff/np.sum(weights)
    if np.isnan(avgDiff):
        print ("found Nan")
    return avgDiff


def measureSimilarity(var, signal, df, nanLabel):
    [samplesNum, columnsNum] = df.shape
    distanceValues = np.empty((samplesNum*WAVE_NUM,3))
    distanceValues[:] = np.nan
    windowLength = len(signal[0])
    counter= 0
    for index in tqdm(range(0, len(df))):
        for w in range(8,15):
            cols= []
            for i in range(w-(windowLength-1),w+1):
                col = "{}_n_{}".format(var,i)
                cols.append(col)
            seq = np.array(df.ix[index, cols])
            seqLabel = np.array(nanLabel.ix[index, cols])
            diff = computeDistance(seq, signal[0], seqLabel, weights=signal[1])
            distanceValues[counter,:]=  [int(index), int(w), diff]
            counter = counter+1     
    return distanceValues


def measureSimilarity_efficient(var, signal, df, nanLabel, place_holder):
    distanceValues, S, SL = [place_holder[0].copy(),place_holder[1].copy(), place_holder[2].copy()] 
    winLen = len(signal[0])
    signalSeq= signal[0]
    weights = signal[1]
    seqLabel = np.zeros((winLen,), dtype=int)
    alpha =0.3
    S = S.reshape(S.shape[0], 1, S.shape[1]) 
    SL = SL.reshape(SL.shape[0], 1, SL.shape[1])
    diff = np.abs(S- signalSeq )
    isNan = np.logical_or(SL, seqLabel)
    penalizedDiff = (1-isNan)*diff + (isNan)*((1-2*alpha)*diff+alpha)
    penalizedDiff =  np.isnan(penalizedDiff)*np.ones(penalizedDiff.shape, dtype=int) + (1 - np.isnan(penalizedDiff))*penalizedDiff
    weightedDiff  = penalizedDiff * weights 
    costSum =np.sum(weightedDiff,axis=2)
    varCost = costSum / np.sum(weights) 
    D = varCost.reshape(len(distanceValues))
    where_are_NaNs = np.isnan(D)
    D[where_are_NaNs] = 1
    distanceValues[:,2]= D
    return distanceValues


def checkIsNan(x):
   return np.isnan(x)


def identifyTreatmentGroup(var, signal, df, nanLabel):
    D = measureSimilarity(var, signal, df, nanLabel)
    

def MAD(data, axis=None):
    res= np.median(np.absolute(data - np.median(data, axis)), axis)
    return res/0.67


def fixReverseDir(df):
    for var in (reverseVars):
        dfs=[]
        for i in range(1,8):
            col = "{}_{}".format(var,i)
            dfs.append(pd.DataFrame( {var: df[col]}))
        mergedDf = pd.concat(dfs)
        maxValue = mergedDf[var].max()
        for i in range(1,8):
            col = "{}_{}".format(var,i)
            df[col] = maxValue - df[col]
    return df


def preprocess(df):
    df = fixReverseDir(df)
    df= normalizeData(df)
    df= expandData(df)
    nanLabel = df.apply(checkIsNan) 
    df = interpolate(df)
    return df, nanLabel


def detectOutliers(distanceInfo, nanLabel, var, string, L=3):
    isKnown = []
    for i in range(0,len(distanceInfo)):
        index = int(distanceInfo[i,0])
        w = int(distanceInfo[i,1])
        col= "memIndex_{}".format(w)
        isKnown.append(not nanLabel.loc[index,col])
    isKnown = np.array(isKnown)
    D = distanceInfo[:,2].copy()
    outliers = D < 0.1
    outliers =  np.logical_and(outliers, isKnown)
    outliersIndex = np.where(outliers)[0]
    print ("passed samples: {}".format(len(outliersIndex)))
    UPPER_LIMIT=2000
    if (len(outliersIndex)>UPPER_LIMIT):
        idx = np.argpartition(D[outliersIndex], UPPER_LIMIT)[:UPPER_LIMIT]
        outliersIndex = outliersIndex[idx]
    print ("samples size after pruning: {}".format(len(outliersIndex)))
    return outliersIndex


def computeAvgDistance2(df, nanLabel, outliersIndexT, outliersIndexC, distanceInfoT, distanceInfoC, varSet, isTargetVar, trtSeq):
        alpha = 0.3
        anchorPoint = (np.where(trtSeq==1)[0][0]-1)
        if (anchorPoint<0):
            anchorPoint=0
        anchorDist = len(trtSeq)- anchorPoint

        anchorPoint2 = (np.where(trtSeq!=2)[0][0])
        anchorDist2 = len(trtSeq)- anchorPoint2   
        
        if (isTargetVar):
            extractLen = anchorDist
            winLen= extractLen-1
            effectiveWeights = np.geomspace(1, (0.5)**(winLen-1) , num=winLen)
        else:
            extractLen = anchorDist2 
            winLen = extractLen
            effectiveWeights = np.ones(winLen)
        
        costSum = np.zeros(shape = (len(outliersIndexT), len(outliersIndexC)))

        for var in (varSet):
            T = np.zeros(shape = (len(outliersIndexT), winLen))
            C = np.zeros(shape = (len(outliersIndexC), winLen))
            TL = np.zeros(shape = (len(outliersIndexT), winLen))
            CL = np.zeros(shape = (len(outliersIndexC), winLen))
            
            for i in range(0,len(outliersIndexT)):
                seqs =extractSeq(df, nanLabel, var, distanceInfoT[outliersIndexT[i],0], distanceInfoT[outliersIndexT[i],1], isTargetVar, length = extractLen)
                T[i,:] = seqs[0][:winLen] # to discard memIndex for the current weight (right most one)
                TL[i,:] = seqs[1][:winLen]

            for j in range(0,len(outliersIndexC)):
                seqs =extractSeq(df, nanLabel, var, distanceInfoC[outliersIndexC[j],0], distanceInfoC[outliersIndexC[j],1], isTargetVar, length = extractLen)
                C[j,:] = seqs[0][:winLen]
                CL[j,:] = seqs[1][:winLen]
            
            T = T.reshape(T.shape[0], 1, T.shape[1])
            TL = TL.reshape(TL.shape[0], 1, TL.shape[1])
            diff = np.abs(T-C)  
            isNan = np.logical_or(TL,CL).astype(int)
            penalizedDiff = (1-isNan)*diff + (isNan)*((1-2*alpha)*diff+alpha)           
            penalizedDiff =  np.isnan(penalizedDiff).astype(int)*np.ones(penalizedDiff.shape, dtype=int) + (1 - np.isnan(penalizedDiff).astype(int))*np.nan_to_num(penalizedDiff)
            weightedDiff  = penalizedDiff * effectiveWeights
            aggregatedCost =np.sum(weightedDiff,axis=2)
            varCost = aggregatedCost / np.sum(effectiveWeights)
            costSum = costSum + varCost

        avgCost= costSum/float(len(varSet))
        return avgCost


def computeAvgDistance2_RBD(df, nanLabel, outliersIndexT, outliersIndexC, distanceInfoT, distanceInfoC, varSet, isTargetVar, trtSeq):
        alpha = 0.3
        anchorPoint = (np.where(trtSeq==1)[0][0]-1)
        if (anchorPoint<0):
            anchorPoint=0
        anchorDist = len(trtSeq)- anchorPoint

        anchorPoint2 = (np.where(trtSeq!=2)[0][0])
        anchorDist2 = len(trtSeq)- anchorPoint2   
        if (isTargetVar):
            extractLen = anchorDist
            winLen= extractLen-1
            effectiveWeights = list(np.arange(100,-1,-100/(winLen)))[:winLen]
        else:
            extractLen = anchorDist2 
            winLen = extractLen
            effectiveWeights = np.ones(winLen)
        
        totalNum = len(outliersIndexT)+len(outliersIndexC)
        costSum = np.zeros(shape = (totalNum,totalNum))
        for var in (varSet):
            T = np.zeros(shape = (totalNum, winLen))
            C = np.zeros(shape = (totalNum, winLen))
            TL = np.zeros(shape = (totalNum, winLen))
            CL = np.zeros(shape = (totalNum, winLen))    
            ids = []
            for i in range(0,len(outliersIndexT)):
                ids.append((distanceInfoT[outliersIndexT[i],0], distanceInfoT[outliersIndexT[i],1]))

            for j in range(0,len(outliersIndexC)):
                ids.append((distanceInfoC[outliersIndexC[j],0], distanceInfoC[outliersIndexC[j],1]))

            for i, pair in enumerate(ids):
                seqs =extractSeq(df, nanLabel, var, pair[0], pair[1], isTargetVar, length = extractLen)
                T[i,:] = seqs[0][:winLen] # to discard memIndex for the current weight (right most one)
                TL[i,:] = seqs[1][:winLen]
            C = T.copy()
            CL = TL.copy()
            T = T.reshape(T.shape[0], 1, T.shape[1])
            TL = TL.reshape(TL.shape[0], 1, TL.shape[1])
            diff = np.abs(T-C)  
            isNan = np.logical_or(TL,CL).astype(int)
            penalizedDiff = (1-isNan)*diff + (isNan)*((1-2*alpha)*diff+alpha)           
            penalizedDiff =  np.isnan(penalizedDiff).astype(int)*np.ones(penalizedDiff.shape, dtype=int) + (1 - np.isnan(penalizedDiff).astype(int))*np.nan_to_num(penalizedDiff)
            weightedDiff  = penalizedDiff * effectiveWeights
            aggregatedCost =np.sum(weightedDiff,axis=2)
            varCost = aggregatedCost / np.sum(effectiveWeights)
            costSum = costSum + varCost

        avgCost= costSum/float(len(varSet))
        return avgCost


def computeDistanceMatrix2_RBD(df, nanLabel, trtVariable, outliersIndexT, outliersIndexC, distanceInfoT, distanceInfoC, trtSeq):
    trtDist= computeAvgDistance2_RBD(df, nanLabel, outliersIndexT, outliersIndexC, distanceInfoT, distanceInfoC, trtmntVar-set([trtVariable]), False, trtSeq)
    confDist= computeAvgDistance2_RBD(df, nanLabel, outliersIndexT, outliersIndexC, distanceInfoT, distanceInfoC, confoundersVar, False, trtSeq)
    targetDist=computeAvgDistance2_RBD(df, nanLabel, outliersIndexT, outliersIndexC, distanceInfoT, distanceInfoC, targetVar, True, trtSeq)
    C= (trtDist + confDist + 2*targetDist)/4
    return C


def fixPairsOffset(matchedPairs, trtNUM):
    matchedPairs_fixed = []
    for pair in  matchedPairs:
        matchedPairs_fixed.append((pair[0], pair[1]-trtNUM))
    return matchedPairs_fixed


def extractTargetValues(df, matchedPairs, outliersIndexT, outliersIndexC,distanceInfoT, distanceInfoC, var, anchorDist):
    print (anchorDist)  
    memtotT = []
    memtotC = []
    memtotT_prev = []
    memtotC_prev = []
    T_ids= []
    C_ids = []
    conf_T = {}
    conf_C = {}
    confVarSet =  (trtmntVar|confoundersVar|targetVar)
    for var in confVarSet:
        conf_T[var] = [[],[],[],[],[],[],[]]
        conf_C[var] = [[], [], [], [], [], [], []]
    for pair in matchedPairs:
        index, w  = distanceInfoT[outliersIndexT[pair[0]],0], distanceInfoT[outliersIndexT[pair[0]], 1]
        w=int(w)
        index=  int(index)
        T_ids.append((index,w))
        col= "memIndex_n_{}".format(w)
        col_prev = "memIndex_n_{}".format(w-anchorDist+1)
        memtotT.append(df.loc[index, col] - df.loc[index, col_prev])
        memtotT_prev.append(df.loc[index, col_prev])
        for confVar in confVarSet:
            for waveOffset in range(0,7):
                col_conf = "{}_n_{}".format(confVar,w-waveOffset)
                conf_T[confVar][waveOffset].append(df.loc[index, col_conf])
    for pair in matchedPairs:
        index, w  = distanceInfoC[outliersIndexC[pair[1]],0],distanceInfoC[outliersIndexC[pair[1]], 1]
        w=int(w)
        index=  int(index)
        C_ids.append((index,w))
        col= "memIndex_n_{}".format(w)
        col_prev = "memIndex_n_{}".format(w-anchorDist+1)
        memtotC.append( df.loc[index, col]-df.loc[index, col_prev])
        memtotC_prev.append( df.loc[index, col_prev])
        for confVar in confVarSet:
            for waveOffset in range(0,7):
                col_conf = "{}_n_{}".format(confVar,w-waveOffset)
                conf_C[confVar][waveOffset].append(df.loc[index, col_conf])
    return [memtotC, memtotT, memtotC_prev, memtotT_prev, conf_C, conf_T]


def extractSeq(df, nanLabel, var, index, w, isTargetVar, length):
    s = int(w-(length-1))
    e = int(w+1)
    cols= []
    for k in range(s,e):
        col = "{}_n_{}".format(var,k)
        cols.append(col)
    seq = np.array(df.loc[index, cols])
    seqLabel = np.array(nanLabel.loc[index, cols])
    return (seq, seqLabel)


def getVarDistances(df, nanLabel):
    result= []
    for var in trtmntVar:
        distanceInfoT = measureSimilarity(var, getTreatmentSignal(), df, nanLabel)
        distanceInfoC = measureSimilarity(var, getControlSignal(), df, nanLabel)
        DT= distanceInfoT[:,2]
        DC= distanceInfoC[:,2]
        result.append((DT,DC))
    return result


def drawKmeanDiagram(Data):
    Data= Data.reshape(-1, 1)
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(Data)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def computeDiff(ids, df):
    diffs = []
    for pair in ids:
        index, w= pair.rstrip().split(" ")
        index = int(index)
        w= int(w)
        col = "memIndex_{}".format(w)
        col_prev = "memIndex_{}".format(w-1)
        diffs.append(df.loc[index,col]-df.loc[index,col_prev])
    return diffs


def fetchLEHyps(path):
    hyps = genfromtxt(path, delimiter=',')
    LEHyps = set()
    for i in range(len(hyps)):
        seq = hyps[i,:].astype(int)
        if (seq==1).any():
            strSeq= str(seq)
            pattern = re.sub("[^0-9]", "",strSeq)
            LEHyps.add(pattern)
    return LEHyps


def splitStr(word): 
    return [int(char) for char in word]


def array2id(array):
    strSeq= str(array.astype(int))
    pattern = re.sub("[^0-9]", "",strSeq)
    return pattern


def id2array(id):
    charList = splitStr(id)
    return np.array(charList)


cache={}

def evaluate_RBD_efficient(var, trtSeq, df, nanLabel, place_holder):
    #if trtSeq in cache:
    #    return cache[trtSeq]

    trtSeq = id2array(trtSeq)
    weights = (~(trtSeq==2)).astype(int)
    trtSignal = (trtSeq,weights)
    distanceInfoT = measureSimilarity_efficient(var, trtSignal, df, nanLabel, place_holder)
    outliersIndexT = detectOutliers(distanceInfoT, nanLabel, var, "Treatment")

    anchorPoint = (np.where(trtSeq!=2)[0][0])
    anchorDist = len(trtSeq)- anchorPoint

    ctrlSeq = np.zeros(len(trtSeq))
    ctrlWeights = np.ones(len(trtSeq))
    ctrlWeights[:(len(trtSeq)-anchorDist)]=0
    ctrlSignal = (ctrlSeq, ctrlWeights)

    distanceInfoC = measureSimilarity_efficient(var, ctrlSignal, df, nanLabel, place_holder)
    outliersIndexC = detectOutliers(distanceInfoC, nanLabel, var, "Control")

    if (len(outliersIndexC)==0 or  len(outliersIndexT)==0 ):
        l.acquire()
        with open("searchResult.txt","a") as f:
            f.write("{0} pattern: {1} ; {2}\n".format(var, trtSeq.astype(int), "NA - outlier detection returned zero samples"))
        l.release()
        cache[array2id(trtSeq)] = np.nan
        return np.nan

    C = computeDistanceMatrix2_RBD(df, nanLabel, var, outliersIndexT, outliersIndexC, distanceInfoT, distanceInfoC, trtSeq)
    matchedPairs = performMatching_RBD(C, len(outliersIndexT))
    if (len(matchedPairs)<4):
        l.acquire()
        with open("searchResult.txt", "a") as f:
            f.write("{0} pattern: {1} ; {2}\n".format(var, trtSeq.astype(int), "NA - matching returned less than four samples out of ({},{})".format(len(outliersIndexT), len(outliersIndexC))))
        l.release()
        cache[array2id(trtSeq)] = np.nan
        return np.nan             

    matchedPairs = fixPairsOffset(matchedPairs, len(outliersIndexT))
    targetValues = extractTargetValues(df, matchedPairs, outliersIndexT, outliersIndexC,distanceInfoT, distanceInfoC, var, anchorDist)
    pval = computePValue(targetValues[0], targetValues[1])
    l.acquire()
    with open("searchResult.txt","a") as f:
        f.write("{0} pattern: {1}; pval={2:}; ACE={4: .2f}; n={3:d}\n".format(var, trtSeq.astype(int), pval, len(matchedPairs), np.mean(targetValues[1])- np.mean(targetValues[0])))
    l.release()
    cache[array2id(trtSeq)]= pval
    return pval


def get_place_holder(var, df, nanLabel, signalLength):
    [samplesNum, columnsNum] = df.shape
    distanceValues = np.empty((samplesNum*WAVE_NUM,3))
    distanceValues[:] = np.nan
    winLen = signalLength
    seqLabel = np.zeros((winLen,), dtype=int)
    
    S = np.zeros(shape = (samplesNum*WAVE_NUM, winLen))
    SL = np.zeros(shape = (samplesNum*WAVE_NUM, winLen))
    counter= 0
    for index in tqdm(range(0, len(df))):
        for w in range(8,15):
            seqs= extractSeq(df, nanLabel, var, index, w, False, length = winLen)
            S[counter,:]= seqs[0]
            SL[counter,:]= seqs[1]
            distanceValues[counter,:]=  [int(index), int(w), np.nan]
            counter = counter+1
    print (distanceValues[0,:])
    return [distanceValues, S, SL]


def runHyps_efficient(var, LowE_Path):
    if (os.path.isfile(dfPath) and os.path.isfile(nanLabelPath)):
        df = pd.read_pickle(dfPath)
        nanLabel = pd.read_pickle(nanLabelPath)
    else:
        df = readData()
        df, nanLabel = preprocess(df)
    pVals = {}
    U = fetchLEHyps(LowE_Path)
    U=list(U)
    print ("len(U):{}".format(len(U)))
    signalLength = len(next(iter(U)))
    place_holder = get_place_holder(var, df, nanLabel, signalLength)
    print("evaluating hyps:")
    pfunc= partial(worker, var=var, df=df, nanLabel=nanLabel, place_holder=place_holder)
    pool = Pool(10)
    with pool as p:
        p.map(pfunc, U)
    pool.close()
    pool.join()

    
def runHypsForAllVars(lowE_Path):
   for var in ["heactb","scorg05","scfrda","scfrdg","scfrdm","heacta", "heactc", "scorg03","scorg06", "scorg07"]:
      print ("var: {}".format(var))
      runHyps_efficient(var, lowE_Path)


def worker(trtSeq, df, nanLabel, place_holder, var):
   evaluate_RBD_efficient(var=var, trtSeq=trtSeq, df=df, nanLabel=nanLabel, place_holder=place_holder)


# if __name__ == "__main__":
    


