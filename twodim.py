"""
twodim.py

Kyle Kovacik

Model-Based Clustering Using only Two Dimensions
- Used to test visually to help when doing multiple dimensions

"""

import numpy as np
import pandas as pd
import random
import math

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

import json
import copy
random.seed(a= "slugcat2", version=2)


features, clusters = make_blobs(n_samples = 1000,n_features = 2, centers = 3, cluster_std = 0.5,shuffle = True)


dishesPath = './data/dishes.csv'
trainPath = './data/user_ratings_train.json'
testPath = './data/user_ratings_test.json'

dishes = pd.read_csv(dishesPath)
dishes = dishes.to_numpy()
dishNp = {}

# Generate a Dictionary for Dishes
counter = 0
for dish in features:

    temp = np.zeros(len(dish))
    for i in range(len(dish)):
        temp[i] = dish[i]
        
    dishNp[counter] = np.array([temp])
    counter += 1


with open(trainPath) as train_json:
    train = json.load(train_json)

#Build Dictionarys for Training Data
trainDict = {}
averageRatings = {}
for user in train:
    trainDictInner = {}
    
    average = 0
    for dishRating in train[user]:
        trainDictInner[dishRating[0]] = dishRating[1]
        average += dishRating[1]

    trainDict[user] = trainDictInner
    averageRatings[user] = average / len(trainDict[user])

#Build Rating Vectors for Users
ratingVectors = {}
for user in train:
    ratingVector = []
    for dish in dishes:
        if dish[0] in trainDict[user]:
            ratingVector.append(trainDict[user][dish[0]])
        else:
            ratingVector.append(0)

    ratingVectors[user] = np.array(ratingVector)
#print(ratingVectors["0"])

with open(testPath) as test_json:
    test = json.load(test_json)

#Build Dictionary for Testing Data
testDict = {}
for user in test:
    testDictInner = {}
    for dishRating in test[user]:
        testDictInner[dishRating[0]] = dishRating[1]

    testDict[user] = testDictInner



"""
    Model Based Collaborative Filtering
    - Soft K - Means Clustering


"""


"""

Task 1 - Predict User Rating From 1-5

- Using Mean Absolute Error (MAE) to evaluate

"""
# Soft K-Mean

#Randomly Select k documents to be the initial cluster centers

# Gaussian Mixture Models with Expectation Maximization????



# Set Number of Clusters
k = 3 # Usually 10



# Number of Ingredients
ingredients = 2


# Guess k gaussians - means and covariance
clusterMean = []
clusterVariance = []

for i in range(k):
    temp2 = []
    for j in range(ingredients):
        # Build covariance diagonal of 1s
        row = []
        for n in range(j):
            row.append(0)
        row.append(1)
        for n in range(ingredients - j - 1):
            row.append(0)
        temp2.append(row)
    
    #Choose a random dish to represent guassian mean
    value = random.randint(0, len(dishNp))
    test = dishNp[value]
    clusterMean.append(test)
    
    clusterVariance.append(np.array(temp2))



# Used to Apply Noise to Covariance to avoid Singularity
varianceNoise =  clusterVariance[0] * math.pow(10, -6)

#Calculate Probability 
def probabilityModel (x, mean, deter, inv):
    determinant = deter
    norm_const = 1.0 / math.sqrt(2 * np.pi * determinant)
    error = x - mean
    result = math.pow(math.e, -0.5 * np.linalg.multi_dot([error, inv, error.T]))

    return norm_const * result



prevClusters = clusterMean

# Begin Iterating until Convergence (Or 1000 Iterations)
for iteration in range(1000):
    prevprevClusters = copy.deepcopy(prevClusters)
    prevClusters = copy.deepcopy(clusterMean)

    # To Save Time Pre-calculate the Inverse Covariance and Determinant for Each Cluster
    inv = {}
    deter = {}
    for i in range(k):
        deter[i] = np.linalg.det(clusterVariance[i])
        if (deter[i] == 0):
            clusterVariance[i] = clusterVariance[i] + varianceNoise
            deter[i] = np.linalg.det(clusterVariance[i])
        inv[i] = np.linalg.inv(clusterVariance[i])

    probs = {}
    #To Save Time Calculate All Probabilities
    counter = 0
    for dish in features:
        
        prob = {}
        for i in range(k):
            prob[i] = probabilityModel(dishNp[counter], clusterMean[i], deter[i], inv[i])

        probs[counter] = prob
        
        counter += 1

    
    maxProb = {}
    counter2 = 0
    for dish in features:
        maxProb[counter2] = [0, 0, 0] #Cluster, Probability
        counter2 += 1

    #Update Mean and Covariance for Each Cluster
    for cluster in range(k):

        mean = np.zeros(ingredients)
        variance = np.zeros((ingredients, ingredients))

        counter1 = 0
        for dish in features:

            #Get ProbXiC for Each Cluster
            probSum = 0
            for i in range(k):
                probSum += probs[counter1][i] * (1.0/k)

            # Get probCXi 
            probCX = (probs[counter1][cluster] * (1.0/k)) / (probSum)
       

            maxProb[counter1][cluster] = probCX

            probStuff = (probCX / ( (1.0/k) * len(features) ))

            # Add to Mean Sum
            mean = np.add(mean, (probStuff * dishNp[counter1]))

            # Add to Variance Sum
            error = dishNp[counter1] - mean
            variance = np.add(variance, probStuff * np.matmul(error.T, error))
            counter1 += 1

        # Set New Mean and Variance for Each Cluster
        clusterMean[cluster] = mean
        clusterVariance[cluster] = variance


    # Every 5 Iterations, Check if models have converged
    if (iteration % 1) == 0:
        print("-------------------------------------")
        print("Iteration: %d" %(iteration))
        
        for kcluster in range(k):
            print("Cluster %d: %f" %(kcluster, np.linalg.norm(prevClusters[kcluster] - clusterMean[kcluster])))

        c = np.empty((0,4), float)
        for i in range(len(maxProb)):
            
            c = np.append(c, np.array([[maxProb[i][0], maxProb[i][1], maxProb[i][2], 1.0]]), axis = 0)


        x = features[:,0]
        y = features[:,1]
        m = np.ones(len(features)) * 5
    

        
        for cluster in range(k):
            x = np.append(x,prevClusters[cluster][0][0])
            y = np.append(y,prevClusters[cluster][0][1])
            c = np.append(c, np.array([[0,0,0, 1.0]]), axis = 0)
            c[len(c)-1][cluster] = 1.0
            m = np.append(m, 500)
            
        #Check if all models have converged
        counter2 = 0
        for kcluster in range(k):
            if ((np.linalg.norm(prevClusters[kcluster] - clusterMean[kcluster])) < 0.0005):
                counter2 = counter2 + 1
        
        # All models are less than 0.005
        if (counter2 == k):
            break



# Now have clusterMean and clusterVariance for k to predict off


# Function to Generate Probability a dish belongs in a cluster
def probability(dishNumber, cluster, clusterMean, clusterVariance):
    # Sum probabilities for all Clusters
    prob = 0
    
    probSum = 0
    for i in range(len(clusterMean)):
        covariance = clusterVariance[i]

        determinant = np.linalg.det(covariance)
        if (determinant == 0):
            covariance += varianceNoise
            determinant = np.linalg.det(covariance)

        norm_const = 1.0 / math.sqrt(2 * np.pi * determinant)
        error = dishNp[dishNumber] - clusterMean[i]
        result = math.pow(math.e, -0.5 * np.linalg.multi_dot([error, np.linalg.inv(clusterVariance[i]), error.T]))
        
        if (i == cluster):
            prob = norm_const * result
        
        probSum += (norm_const * result) * (1.0/k)

    # Find proportion of probability to cluster over all clusters
    probCX = (prob * (1.0/k)) / (probSum)
    
    return probCX


#Test Against 3 Dishes
total = 0
for cluster in range(k):
    print("------------------------------")
    print("Dish %d | Cluster %d: %f" %(10, cluster, probability(10, cluster, clusterMean, clusterVariance)))
    print("Dish %d | Cluster %d: %f" %(1, cluster, probability(1, cluster, clusterMean, clusterVariance)))
    print("Dish %d | Cluster %d: %f" %(24, cluster, probability(24, cluster, clusterMean, clusterVariance)))
    print("Dish %d | Cluster %d: %f" %(150, cluster, probability(150, cluster, clusterMean, clusterVariance)))

x = features[:,0]
y = features[:,1]
m = np.ones(len(features)) * 5

c = np.empty((0,4), float)
for i in range(len(maxProb)):   
    c = np.append(c, np.array([[maxProb[i][0], maxProb[i][1], maxProb[i][2], 1.0]]), axis = 0)

for cluster in range(k):
    x = np.append(x,prevClusters[cluster][0][0])
    y = np.append(y,prevClusters[cluster][0][1])
    c = np.append(c, np.array([[0,0,0, 1.0]]), axis = 0)
    c[len(c)-1][cluster] = 1.0
    m = np.append(m, 500)

def addGraph(dishNumber, x, y, c, m):
    x = np.append(x, dishNp[dishNumber][0][0])
    y = np.append(y, dishNp[dishNumber][0][1])
    c = np.append(c, np.array([[probability(dishNumber, 0, clusterMean, clusterVariance), probability(dishNumber, 1, clusterMean, clusterVariance), probability(dishNumber, 2, clusterMean, clusterVariance), 1.0]]) , axis = 0)
    m = np.append(m, 1000)

    return x, y, c, m

x, y, c, m = addGraph(200, x, y, c, m)
x, y, c, m = addGraph(300, x, y, c, m)

plt.scatter(x, y, color=c, s=m)
plt.show()