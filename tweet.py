import sys
import json
import re
import logging
import numpy as np
import pandas as pd

logging.basicConfig()
logger = logging.getLogger('tweet')
logger.setLevel(logging.INFO)

# read tweet and convert them to a list including text and id
def frame(jsonpath):
    logger.info('Starting to read Tweets.json')
    frame = []
    readin = open(jsonpath)
    for line in readin:
        row=[]
        linein = json.loads(line)
        text = linein["text"]
        row.append(text)
        idin = linein["id"]
        row.append(idin)
        frame.append(row)
    logger.info('Finishing to read Tweets.json')

    logger.info('Starting to parse Tweets.json')
    for i in range(len(frame)):
        frame[i][0]=re.sub('[@#]\w*','',frame[i][0])
        frame[i][0]=re.sub(':','',frame[i][0])
        frame[i][0]=re.sub('RT','',frame[i][0])
        frame[i][0]=re.sub('\w*//?\w*','',frame[i][0])
        frame[i][0]=frame[i][0].split(" ")
    logger.info('Finishing to parse Tweets.json')
    return frame

# input the initial centroids and their contents
def centroid():
    logger.info('Starting to initializing centroid')
    center=pd.read_csv("initial.csv",header=None)
    center=center[0]
    centroid=[]
    for point in center:
        row=[]
        row.append(point)
        for data in frame:
            if point == data[1]:
                row.append(data[0])
        centroid.append(row)
    logger.info('Starting to initializing centroid')
    return centroid

# calculate the Jaccard distance
def jaccarddistance(list1, list2):
    U=len(set(list1).union(set(list2)))
    I=len(set(list1).intersection(set(list2)))
    dist=(U-I)/U
    return dist
    

def k_mean(data, centroid, kclusters=25):
    logger.info('Starting clustering')
    data=np.array(data)
    # add one column to store assigned clusters
    col1=np.zeros((data.shape[0],1))
    data=np.append(data,col1,axis=1)
    col2=np.zeros((data.shape[0],1))
    data=np.append(data,col2,axis=1)

    for n in range(20):
        for i in range(data.shape[0]):
            min=sys.maxsize
            for j in range(kclusters):
                dist=jaccarddistance(data[i][0],centroid[j][1])
                if min>dist:
                    min=dist
                    data[i][2]=centroid[j][0]
                    data[i][3]=min
        
        for i in range(kclusters):
            subdata=[]
            for j in range(data.shape[0]):
                row=[]
                if data[j][2]==centroid[i][0]:
                    row.append(data[j][0])
                    row.append(data[j][1])
                    subdata.append(row)
            min=sys.maxsize
            for j in range(len(subdata)):
                total=0
                for k in range(len(subdata)):
                    dist=jaccarddistance(subdata[j][0], subdata[k][0])
                    total +=dist
                if total<min:
                    min=total
                    centroid[i][0]=subdata[j][1]
    logger.info('Starting clustering')
    return data,centroid    

frame = frame(sys.argv[1])
centroid = centroid()
data,centroid=k_mean(frame,centroid)
data=pd.DataFrame(data).iloc[:,1:]
data.columns=['id','cluster_id','distance']
result=data.sort_values('cluster_id')
result.to_csv(sys.argv[2])