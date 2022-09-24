#!/usr/bin/env python
# coding: utf-8

# In[12]:


# coding: utf-8
import re
import glob
import math
import csv
import sys
import os
import struct
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from obspy import UTCDateTime, Stream, Trace
from tqdm.notebook import tqdm

import seisbench
import seisbench.models as sbm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def unpackAfile(infile):

# == opening Afile ==
    b = os.path.getsize(infile)
    FH = open(infile, 'rb')
    line = FH.read(b)
    fileHeader = struct.unpack("<4s3h6bh6s", line[0:24])

    fileLength = fileHeader[3]
    port = fileHeader[10]
    # FirstStn = fileHeader[11][0:4].decode('ASCII').rstrip()
# =================================Header=================================

    portHeader = []
    for i in range(24, port * 32, 32):
        port_data = struct.unpack("<4s4s3sbh2b4s12b", line[i:i+32])
        portHeader.append(port_data)

# =================================Data===================================

    dataStartByte = 24+int(port)*32
    dataPoint = 3*int(port)*int(fileLength)*100
    times = int(port)*3*4
    data = []

    data = struct.unpack("<%di" % dataPoint, line[dataStartByte:dataStartByte + dataPoint*4])


    portHeader = np.array(portHeader)
    data = np.array(data)
    idata = data.reshape((3,port,fileLength*100),order='F')

#== write to obspy Stream --
    sttime = UTCDateTime(fileHeader[1], fileHeader[4], fileHeader[5], fileHeader[6], fileHeader[7], fileHeader[8], fileHeader[2])
    npts = fileHeader[3]*fileHeader[9]
    samp = fileHeader[9]
    afst = Stream()
    
    for stc in range(fileHeader[10]):
        stn = portHeader[stc][0].decode('ASCII').rstrip()
        instrument = portHeader[stc][1].decode('ASCII').rstrip()
        loc = '0'+str(portHeader[stc][6].decode('ASCII'))
        net = str(portHeader[stc][7].decode('ASCII')).rstrip()
        GPS = int(portHeader[stc][3])
        
        # remove GPS unlock or broken station
        if ( GPS == 1 or GPS == 2 ):
            chc = 0
            if instrument == 'FBA':
                chc = 1
            elif instrument == 'SP':
                chc = 4
            elif instrument == 'BB':
                chc = 7
            
            for ch in range(3):
                chn = 'Ch'+str(chc+ch)
                
                stats = {'network': net, 'station': stn, 'location': loc,
                        'channel': chn, 'npts': npts, 'sampling_rate': samp,
                        'starttime': sttime}
                
                data = np.array(idata[ch][stc], dtype=float)
                sttmp = Stream([Trace(data=data, header=stats)])
                afst += sttmp

    return afst


# In[3]:


def unpackPfile(infile):
    
    with open(infile) as f:
        lines = f.readlines()
    
    tmp = lines[0]
    year = int(tmp[1:5])
    month = int(tmp[5:7])
    day = int(tmp[7:9])
    hour = int(tmp[9:11])
    minute = int(tmp[11:13])
    sec = float(tmp[13:19])

    dt = datetime(year,month,day,hour,minute,int(sec//1),int(sec%1 * 1000000))
    mag = float(tmp[40:44])

    pfile_info = {}
    pfile_info["ori_time"] = dt
    pfile_info["mag"] = mag

    intensity = {}
    arrival_time_P = {}
    arrival_time_S = {}
    weighting = {}
    pga = {}
    for i in lines[1:]:
        sta = i[:5].strip() # strip 去掉左右空格
        weighting[sta] = int(float(i[35:39]))
        if i[76:77]==" ":
            intensity[sta] = int(0)
        else:
            intensity[sta] = int(i[76:77])
        pga[sta] = float(i[78:83])
        arrival_time_P[sta] = pfile_info["ori_time"].replace(minute=int(i[21:23]),second=0,microsecond=0) + timedelta(seconds=float(i[23:29]))
        arrival_time_S[sta] = pfile_info["ori_time"].replace(minute=int(i[21:23]),second=0,microsecond=0) + timedelta(seconds=float(i[39:45]))
        
    pfile_info["intensity"] = intensity
    pfile_info["arrival_time_P"] = arrival_time_P
    pfile_info["arrival_time_S"] = arrival_time_S
    pfile_info["weighting"] = weighting
    pfile_info["pga"] = pga
    
    return pfile_info


# In[4]:


def network_filter(Afile,net):
    for i in Afile:
        if i.stats.network not in net:
            Afile.remove(i)
    return Afile


# In[5]:


#只針對測站網 "BB","BH","SMT","OBS","CSMT","BATS","YMS" 做修改
def RSD_transform(data):
    for i in data:
        net = i.stats.network
        loc = i.stats.location
        chn = i.stats.channel
        
        if loc == "01":
            if net == "BB" or "BATS" or "YMS":# BB, BATS, YMS
                if chn == "Ch1" or chn == "Ch2" or chn == "Ch3":
                    if chn == "Ch1":
                        chn = "HLZ"
                    elif chn == "Ch2":
                        chn = "HLN"
                    elif chn == "Ch3":
                        chn = "HLE" 
                    loc = "11"    
                elif chn == "Ch7" or chn == "Ch8" or chn == "Ch9":
                    
                    if chn == "Ch7":
                        chn = "HHZ"
                    elif chn == "Ch8":
                        chn = "HHN"
                    elif chn == "Ch9":
                        chn = "HHE" 
                    loc = "10"    
            if net == "SMT" or "CSMT":# SMT CSMT
                
                if chn == "Ch1" or chn == "Ch2" or chn == "Ch3":
                    if chn == "Ch1":
                        chn = "HLZ"
                    elif chn == "Ch2":
                        chn = "HLN"
                    elif chn == "Ch3":
                        chn = "HLE"              
                if chn == "Ch4" or chn == "Ch5" or chn == "Ch6":
                    if chn == "Ch4":
                        chn = "EHZ"
                    elif chn == "Ch5":
                        chn = "EHN"
                    elif chn == "Ch6":
                        chn = "EHE"
                loc = "10"  
            if net == "BH":
                if chn == "Ch1":
                    chn = "HNZ"
                elif chn == "Ch2":
                    chn = "HNN"
                elif chn == "Ch3":
                    chn = "HNE"
                loc = "10"
        if loc == "02":
            if net == "SMT":
                if chn == "Ch1" or chn == "Ch2" or chn == "Ch3":
                    if chn == "Ch1":
                        chn = "HLZ"
                    elif chn == "Ch2":
                        chn = "HLN"
                    elif chn == "Ch3":
                        chn = "HLE"              
                if chn == "Ch4" or chn == "Ch5" or chn == "Ch6":
                    if chn == "Ch4":
                        chn = "EHZ"
                    elif chn == "Ch5":
                        chn = "EHN"
                    elif chn == "Ch6":
                        chn = "EHE"
            else:        
                if chn == "Ch1" or chn == "Ch2" or chn == "Ch3":
                    if chn == "Ch1":
                        chn = "HNZ"
                    elif chn == "Ch2":
                        chn = "HN1"
                    elif chn == "Ch3":
                        chn = "HN2"
                        
                if chn == "Ch7" or chn == "Ch8" or chn == "Ch9":
                    if chn == "Ch7":
                        chn = "HHZ"
                    elif chn == "Ch8":
                        chn = "HH1"
                    elif chn == "Ch9":
                        chn = "HH2"
            loc = "00"     
            
        if loc == "03":# OBS
            
            if chn == "Ch1" or chn == "Ch2" or chn == "Ch3":
                if chn == "Ch1":
                    chn = "HN1"
                elif chn == "Ch2":
                    chn = "HN1"
                elif chn == "Ch3":
                    chn = "HN2"       
            if chn == "Ch4" or chn == "Ch5" or chn == "Ch6":
                if chn == "Ch4":
                    chn = "EH1"
                elif chn == "Ch5":
                    chn = "EH2"
                elif chn == "Ch6":
                    chn = "EH3"
            
            if chn == "Ch7" or chn == "Ch8" or chn == "Ch9":
                if chn == "Ch7":
                    chn = "HL1"
                elif chn == "Ch8":
                    chn = "HL2"
                elif chn == "Ch9":
                    chn = "HL3"
            loc = "20"        
                         
        net = "TW"#測站網全改TW
        i.stats.network = net
        i.stats.location = loc
        i.stats.channel = chn
    return data


# In[6]:


#EQTransfomer
def predict_dict(Afile,model):
    count1 = 0
    count2 = 3
    nameCount = 1
    pred_result ={}
    while (count2 <= Afile.count()):
        tmp_Afile = Afile[count1:count2]
        tmp_preds = model.annotate(tmp_Afile)
        tmp_picks = model.classify(tmp_Afile) 
        
        tmp_picks_dict = {}
        tmp_data ={}
        
        tmp_picks_dict["P_S"] = [i.__dict__ for i in tmp_picks[0]]
        tmp_picks_dict["D"] = [i.__dict__ for i in tmp_picks[1]]
        
        for i in tmp_picks_dict["P_S"]:
            tmp_value = round(i["peak_value"]*100)
            if tmp_value >= 0 and tmp_value < 20:
                tmp_weighting = 4
            elif tmp_value >= 20 and tmp_value < 40:
                tmp_weighting = 3
            elif tmp_value >= 40 and tmp_value < 60:
                tmp_weighting = 2
            elif tmp_value >= 60 and tmp_value < 80:
                tmp_weighting = 1
            elif tmp_value >= 80 and tmp_value < 100:
                tmp_weighting = 0
            i["weighting"] = tmp_weighting
        
        tmp_data["stream"] = tmp_Afile
        tmp_data["preds"] = tmp_preds
        tmp_data["picks"] = tmp_picks_dict
        
        streamName = "stream" + str(nameCount)
        pred_result[streamName]=tmp_data
        
            
        ###########
        nameCount += 1
        count1 += 3
        count2 += 3
        
    return pred_result


# In[7]:


def plot_pred(stream,preds,wlength):
    color_dict = {"P": "C0", "S": "C1", "Detection": "C2"}
    for trace in stream:
        for s in range(0,int(trace.stats.endtime - trace.stats.starttime),wlength):
            t0 = trace.stats.starttime + s
            t1 = t0 + wlength
            subtr = trace.slice(t0,t1)
        
            subpreds =preds.slice(t0,t1)
            offset = subpreds[0].stats.starttime - subtr.stats.starttime
            
            
            fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True, gridspec_kw={'hspace' : 0.05, 'height_ratios': [2, 1]})
            for pred_tr in subpreds:
                model, pred_class = pred_tr.stats.channel.split("_")
                if pred_class == "N":
                        # Skip noise traces
                        continue
                c = color_dict[pred_class]
                ax[1].plot(offset + pred_tr.times(), pred_tr.data, label=pred_class, c=c)
                
            ax[1].set_ylabel(model)
            ax[1].legend(loc=2)
            ax[1].set_ylim(0, 1.1)
            ax[1].set_xlabel('Time [s]')
            
            #波型正規化使用除以波型最大值
            ax[0].plot(subtr.times(), subtr.data / np.amax(subtr.data), 'k', label=subtr.stats.channel)
            ax[0].set_xlim(0, wlength)
            ax[0].set_ylabel('Normalised Amplitude')
            ax[0].legend(loc=2)
            plt.show()


# In[8]:


def preds_arrivaltime(predict_dict):
    P_S_arrivaltime = {}
    for i in range(len(predict_dict)):
        P_S = predict_dict["stream"+str(i+1)]["picks"]["P_S"]
        if P_S == []:
            continue
        tmp_sta = P_S[0]["trace_id"].split(".")[1]
        P_S_arrivaltime[tmp_sta] = {"P":[],"S":[]}
        tmp_P_arrivaltime = []
        tmp_S_arrivaltime = []
        primary_P = []
        primary_S = []
        for j in P_S:
            if j["phase"] == "P":
                tmp_P_arrivaltime.append([j['peak_time'],j['weighting']])
        for k in P_S:
            if k["phase"] == "S":
                tmp_S_arrivaltime.append([k['peak_time'],k['weighting']])
                
        if tmp_P_arrivaltime !=[]:
            primary_P = min(tmp_P_arrivaltime)
        if tmp_S_arrivaltime !=[]:
            primary_S = min(tmp_S_arrivaltime)
        
        P_S_arrivaltime[tmp_sta]["P"] = primary_P
        P_S_arrivaltime[tmp_sta]["S"] = primary_S
        
    return P_S_arrivaltime


# In[9]:


#P_file_header
def make_P_file(pred_arrivaltime, path):
    ori_time = min([pred_arrivaltime[i][j][0] for i in pred_arrivaltime for j in pred_arrivaltime[i] if pred_arrivaltime[i][j] != []])
    ori_time_year = str(ori_time.year)
    ori_time_month = str(ori_time.month)
    ori_time_day = str(ori_time.day)
    ori_time_hour = str(ori_time.hour)
    ori_time_minute = str(ori_time.minute)
    ori_time_list = [ori_time_year, ori_time_month, ori_time_day, ori_time_hour, ori_time_minute]
    
    for ind,ele in enumerate(ori_time_list):#將只是個位數的十位數部分補空格
        s = " "
        if int(ele)<10:
            ori_time_list[ind] = s + ele
            
    Pfile_header = " "
    for i in ori_time_list:
        Pfile_header += i
        
    unexpected_header = "  0.0000 0.0000000.00  0.000.0000  0.0 000.00 0.0 0.0"
    Pfile_header += unexpected_header
    #Pfile_content
    Pfile_content = []
    for sta in pred_arrivaltime:
        P_sec, S_sec = ["",""]
        P_weighting, S_weighting = [0,0]
        for P_S in pred_arrivaltime[sta]:
            weighting = 0
            microsec = "00"
            if pred_arrivaltime[sta][P_S] != []:
                weighting = pred_arrivaltime[sta][P_S][1]
                microsec = str(pred_arrivaltime[sta][P_S][0].microsecond)[:2]
                if microsec == "0":
                    microsec = "00"
                tmp_arrivaltime = pred_arrivaltime[sta][P_S][0]
                sec = pred_arrivaltime[sta][P_S][0].second
                counter = pred_arrivaltime[sta][P_S][0].minute - ori_time.minute
                for i in range (counter):
                    sec += 60
            else:
                sec = 0
            for j in range(3-len(str(sec))):
                s = " "
                sec = s + str(sec)
            if P_S == "P":
                P_sec = f"{sec}.{microsec}"
                P_weighting = str(weighting)
            else:
                S_sec = f"{sec}.{microsec}"
                S_weighting = str(weighting)
        for k in range(4-len(sta)):
            s = " "
            sta += s 
        finall_str = f" {sta}  00.0   0   0  {ori_time_list[4]}{P_sec} 0.00 {P_weighting}.00{S_sec} 0.00 {S_weighting}.00 0.00 0.00 0.00 0.00 0   0.0"
        Pfile_content.append(finall_str)
    #write_to_Pfile
    file_year = ori_time_year[2:4]
    if int(ori_time_year) >= 2000:
        file_month = str(int(ori_time_month) + 12)
    file_name_list = [file_month, ori_time_day, ori_time_hour, ori_time_minute, file_year]
    for ind,ele in enumerate(file_name_list):
        s = "0"
        if int(ele)<10:
            file_name_list[ind] = s + ele
    with open(path + f"\\{file_name_list[0]}{file_name_list[1]}{file_name_list[2]}{file_name_list[3]}.P{file_name_list[4]}","w+") as fp:
        fp.write(Pfile_header)
        fp.write("\n")
        for i in Pfile_content:
            fp.write(i)
            fp.write("\n")


# In[10]:


def diff_arravialtime_P(pred_arrivaltime,Pfile):
    diff_arravialtime = {}
    for i in pred_arrivaltime.keys():
        for j in Pfile["arrival_time_P"].keys():
            if j == i and pred_arrivaltime[i]["P"] != []:
                if Pfile["arrival_time_P"][j].second == 0:
                    continue
                diff_arravialtime[j] = pred_arrivaltime[i]["P"][0] - UTCDateTime(Pfile["arrival_time_P"][j])
    return diff_arravialtime


# In[11]:


def diff_arravialtime_S(pred_arrivaltime,Pfile):
    diff_arravialtime = {}
    for i in pred_arrivaltime.keys():
        for j in Pfile["arrival_time_S"].keys():
            if j == i and pred_arrivaltime[i]["S"] != []:
                if Pfile["arrival_time_S"][j].second == 0:
                    continue
                diff_arravialtime[j] = pred_arrivaltime[i]["S"][0] - UTCDateTime(Pfile["arrival_time_S"][j])
    return diff_arravialtime


# In[ ]:


def plot_arrival_P(Pjson, info):
    with open(Pjson, "r") as fp:
        all_diff_arrivaltime_P = json.load(fp)
    with open(info,"r") as fp:
        metadata = fp.read().splitlines() 
    
    info = f"Model : {metadata[3][8:]} \nDataset : {metadata[2][10:]} \nPhase : P"
    title = f"{metadata[0][7:]}_{metadata[3][8:]}_{metadata[2][10:]}_P"
    count_sta_P = [ len(all_diff_arrivaltime_P[i]) for i in all_diff_arrivaltime_P]
    event_list = [i for i in all_diff_arrivaltime_P]
    data_list = [list(all_diff_arrivaltime_P[i].values()) for i in all_diff_arrivaltime_P]
    x = [i + 1 for i in range(len(all_diff_arrivaltime_P))]
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 12), sharex=True, gridspec_kw={'hspace' : 0.05, 'height_ratios': [1, 2]})
    
    ax[0].plot(x,count_sta_P, alpha = 0.5)
    for a, b in zip(x, count_sta_P):
        ax[0].text(a, b ,b, ha = "center" , va = "center", fontsize = 16, c = "red",in_layout = True)
    ax[0].grid("True")
    ax[0].set_title(title, fontsize = 30, pad = 20)
    ax[0].set_ylabel("Stations",fontsize = 16 ,labelpad = 20)
    
    ax[1].boxplot(data_list)
    ax[1].set_xticks(x,event_list,rotation = 20)
    ax[1].set_xlabel("No.event", fontsize = 25, labelpad = 20)
    ax[1].set_ylabel("P Time Residual (s)", fontsize = 16,)
    ax[1].set_ylim(-10, 10)
    ax[1].grid("True")
    fig.text(0.15, 0.5,info, fontsize=16,bbox={"facecolor":"white", "alpha":1, "pad":5})
    fig.savefig(f"{title}.png", bbox_inches = "tight", facecolor = "white")


# In[ ]:


def plot_arrival_S(Sjson, info):
    with open(Sjson, "r") as fp:
        all_diff_arrivaltime_S = json.load(fp)
    with open(info,"r") as fp:
        metadata = fp.read().splitlines() 
    
    info = f"Model : {metadata[3][8:]} \nDataset : {metadata[2][10:]} \nPhase : S"
    title = f"{metadata[0][7:]}_{metadata[3][8:]}_{metadata[2][10:]}_S"
    
    count_sta_S = [ len(all_diff_arrivaltime_S[i]) for i in all_diff_arrivaltime_S]
    event_list = [i for i in all_diff_arrivaltime_S]
    data_list = [list(all_diff_arrivaltime_S[i].values()) for i in all_diff_arrivaltime_S]
    x = [i + 1 for i in range(len(all_diff_arrivaltime_S))]
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 12), sharex=True, gridspec_kw={'hspace' : 0.05, 'height_ratios': [1, 2]})
    
    ax[0].plot(x,count_sta_S, alpha = 0.5)
    for a, b in zip(x, count_sta_S):
        ax[0].text(a, b ,b, ha = "center" , va = "center", fontsize = 16, c = "red",in_layout = True)
    ax[0].grid("True")
    ax[0].set_title(title, fontsize = 30, pad = 20)
    ax[0].set_ylabel("Stations",fontsize = 16 ,labelpad = 20)
    
    ax[1].boxplot(data_list)
    ax[1].set_xticks(x,event_list,rotation = 20)
    ax[1].set_xlabel("No.event", fontsize = 25, labelpad = 20)
    ax[1].set_ylabel("S Time Residual (s)", fontsize = 16)
    ax[1].set_ylim(-10, 10)
    ax[1].grid("True")
    fig.text(0.15, 0.5,info, fontsize=16,bbox={"facecolor":"white", "alpha":1, "pad":5})
    fig.savefig(f"{title}.png", bbox_inches = "tight", facecolor = "white")

