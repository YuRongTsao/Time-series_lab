import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import math
from collections import defaultdict
from sklearn import metrics
from scipy.stats import normaltest
import ruptures as rpt
import matplotlib.mlab as mlab

def generateNormalPoints(mean ,std,amount):
    y = norm.rvs(mean,std,size = amount) #random points
    return y

def fitPointsToNormDist(points):
    mean,std = norm.fit(points)
    return mean,std

def cluster(X,k):
    kmeans = KMeans(n_clusters=k).fit(X)
    return  kmeans.predict(X)  # 得每個點的codebook index

def assignKeyValue(dict,key,value):
    dict[key] = [value]

def getKdistribution(X,labels):
    clusteredData = defaultdict(lambda: 0)
    labels = list(labels)

    for i in range(len(labels)):
        clusteredData[labels[i]].append(X[i][1]) if clusteredData[labels[i]]!=0 else assignKeyValue(clusteredData,labels[i],X[i][1])
    
    distributions = [fitPointsToNormDist(values) for key,values in clusteredData.items()]
    return distributions,clusteredData

def getBestK(X):
    scores = defaultdict(lambda: 0)
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
        labels = kmeans.labels_
        scores[k] = metrics.calinski_harabaz_score(X, labels)
    
    return list(scores.keys())[list(scores.values()).index(max(list(scores.values())))]

def getNormProb(datas): #return true if 2 distribution are all normal
    result = np.product([normaltest(data)[1] if len(data)>8 else 0.001 for data in datas ])
    return result

def getBestCutPoints(X):
    cutPoints = [i for i in range(len(list(X)))]
    values = [tuples[1] for tuples in list(X)]
    splits = [[values[:cutPoint],values[cutPoint:]] for cutPoint in cutPoints]
    probs = [getNormProb(datas) for datas in splits]
    bestCutPoint = probs.index(max(probs))
    return bestCutPoint

def cutPointDetection(X):    
    values = np.array([tuples[1] for tuples in list(X)])
    algo = rpt.Pelt(model="normal").fit(values)
    result = algo.predict(pen=10) #result 的最後一項是總共有幾個點
    return result[:-1]


if __name__ == "__main__":
    
    #Q1
    points_1 = generateNormalPoints(0,1,100)
    points_2 = generateNormalPoints(1,1,100)
    
    #draw
    x = np.linspace(1,100,100)

    fig_1 = plt.figure(1,figsize=(10,8))
    fig_1.suptitle('Hw1', fontsize=16)
    
    ax = plt.subplot('211')
    ax.set_title('Q1: Random samples for t = 1~200')
    plt.plot(x,points_1,'ro',alpha = 0.6,markersize = 3,label = 'N(0,1)')

    x = np.linspace(101,200,100)
    plt.plot(x,points_2,'go',alpha = 0.6,markersize = 3,label = 'N(1,1)')

    plt.legend(loc = 'upper right')
    plt.subplots_adjust(hspace = 1)

    #Q2
    new_points = np.append(points_1,points_2)
    mean,std = fitPointsToNormDist(new_points)
    

    #draw
    ax = plt.subplot('212')
    ax.set_title('Q2: Fit 200 points with single normal distribution\n new mean: '+ str(round(mean,2)) + ' new std:' + str(round(std,2)))
        
    mu,sigma = 0,1
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x,mlab.normpdf(x, mu, sigma),'r',alpha = 0.6,linewidth = 2,label = 'N(0,1)')
    mu = 1
    plt.plot(x,mlab.normpdf(x, mu, sigma),'g',alpha = 0.6,linewidth = 2,label = 'N(1,1)')
    plt.plot(x,mlab.normpdf(x, mean, std),'b',alpha = 0.6,linewidth = 2,label = 'N(' + str(round(mean,2)) + ',' + str(round(std,2)) + ')')
    
    plt.legend(loc = 'upper right')

    #Q3
    x = np.linspace(1,200,200)
    X = np.array(list(zip(x, new_points)))
    labels = cluster(X,2)
    distributions,clusteredData = getKdistribution(X,labels)
    
    #draw
    fig_2 = plt.figure(2,figsize=(10,8))
    fig_2.suptitle('Hw1 - Q3', fontsize=16)

    x = np.linspace(1,len(clusteredData[1]),len(clusteredData[1]))
    points = clusteredData[1]

    ax = plt.subplot('211')
    ax.set_title('Cluster Result (k=2)')
    plt.plot(x,points,'ro',alpha = 0.6,markersize = 3,label = 'cluster 1 ')

    x = np.linspace(len(clusteredData[1])+1,len(clusteredData[1])+len(clusteredData[0]),len(clusteredData[0]))
    points = clusteredData[0]
    plt.plot(x,points,'go',alpha = 0.6,markersize = 3,label = 'cluster 2 ')

    plt.legend(loc = 'upper right')
    plt.subplots_adjust(hspace = 1)

    ax = plt.subplot('212')
    ax.set_title('Distribution of two clusters ')

    #draw distribution
    mu = list(distributions[0])[0]
    sigma = list(distributions[0])[1]
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x,mlab.normpdf(x, mu, sigma),'r',alpha = 0.6,linewidth = 2,label = 'N(' + str(round(mu,2)) + ',' + str(round(sigma,2)) + ')')

    mu = list(distributions[1])[0]
    sigma = list(distributions[1])[1]
    plt.plot(x,mlab.normpdf(x, mu, sigma),'g',alpha = 0.6,linewidth = 2,label = 'N(' + str(round(mu,2)) + ',' + str(round(sigma,2)) + ')')
    
    plt.legend(loc = 'upper right')

    plt.show()


    #Q4 Q5 Q6
    for i in range(20):
        points_1 = generateNormalPoints(0,1,100)
        points_2 = generateNormalPoints(1,1,100)
        x = np.linspace(1,200,200)
        new_points = np.append(points_1,points_2)
        X = np.array(list(zip(x, new_points)))

        #Q4
        bestK = getBestK(X)
        f = open(r'Q4.txt', 'a')
        f.write(str(i+1) + '. best k = ' + str(bestK) + '\n')

        #Q5
        bestCutPoint = getBestCutPoints(X)
        f = open(r'Q5.txt', 'a')
        f.write(str(i+1) + '. best cut point = ' + str(bestCutPoint)+'\n')

        #Q6
        cutPoints = cutPointDetection(X)
        f = open(r'Q6.txt', 'a')
        f.write(str(i+1) + '. number of cut points = ' + str(len(cutPoints)) + ' cut point = '+ str([c for c in cutPoints])+'\n')
   

   
    
