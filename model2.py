# model.py

import numpy as np
import pandas as pd
import random
import math

import json
import copy
random.seed(a= "slugcat2", version=2)

dishesPath = './data/dishes.csv'
trainPath = './data/user_ratings_train.json'
testPath = './data/user_ratings_test.json'

dishes = pd.read_csv(dishesPath)


dishes = dishes.to_numpy()
dishNp = {}
# Generate a Dictionary for Dishes
for dish in dishes:

    temp = np.zeros(len(dish[2:]))
    for i in range(len(dish[2:])):
        temp[i] = dish[i + 2]
        
    dishNp[dish[0]] = np.array([temp])

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
k = 5 # Usually 10



# Number of Ingredients
ingredients = len(dishes[1]) - 2


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
    value = random.randint(0, len(dishes))
    test = dishNp[value]
    clusterMean.append(test)
    
    clusterVariance.append(np.array(temp2))



# Used to Apply Noise to Covariance to avoid Singularity
varianceNoise =  clusterVariance[0] * math.pow(10, -1)

#Calculate Probability 
def probabilityModel (x, mean, deter, inv):
    determinant = deter
    norm_const = 1.0 / math.sqrt(2 * np.pi * determinant)
    error = x - mean
    result = math.pow(math.e, -0.5 * np.linalg.multi_dot([error, inv, error.T]))

    return norm_const * result



prevClusters = clusterMean

# Begin Iterating until Convergence (Or 1000 Iterations)
for iteration in range(1):
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
    for dish in dishes:
        prob = {}
        for i in range(k):
            prob[i] = probabilityModel(dishNp[dish[0]], clusterMean[i], deter[i], inv[i])

        probs[dish[0]] = prob


    #Update Mean and Covariance for Each Cluster
    for cluster in range(k):

        mean = np.zeros(ingredients)
        variance = np.zeros((ingredients, ingredients))

        for dish in dishes:

            #Get ProbXiC for Each Cluster
            probSum = 0
            for i in range(k):
                probSum += probs[dish[0]][i] * (1.0/k)


            # Get probCXi 
            probCX = (probs[dish[0]][cluster] * (1.0/k)) / (probSum)
            probStuff = (probCX / ( (1.0/k) * len(dishes) ))

            # Add to Mean Sum
            mean = np.add(mean, probStuff * dishNp[dish[0]])

            # Add to Variance Sum
            error = dishNp[dish[0]] - mean
            variance = np.add(variance, probStuff * np.matmul(error.T, error))

        # Set New Mean and Variance for Each Cluster
        clusterMean[cluster] = mean
        clusterVariance[cluster] = variance


    # Every 5 Iterations, Check if models have converged
    if (iteration % 1) == 0:
        print("-------------------------------------")
        print("Iteration: %d" %(iteration))
        for kcluster in range(k):
            print("Cluster %d: %f" % (kcluster, np.linalg.norm(prevClusters[kcluster] - clusterMean[kcluster])))
        
        #Check if all models have converged
        count = 0
        for kcluster in range(k):
            if ((np.linalg.norm(prevClusters[kcluster] - clusterMean[kcluster])) < 0.0005):
                count = count + 1
        
        # All models are less than 0.005
        if (count == k):
            break



# Now have clusterMean and clusterVariance for k to predict off

"""
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
        
        probSum += (norm_const * result) * (1.0 / k)

    # Find proportion of probability to cluster over all clusters
    probCX = (prob * (1.0/k)) / (probSum)
    
    return probCX
"""

# Build Final Dish Probability Dictionary

finalInv = {}
finalDeter = {}
for i in range(k):
    finalDeter[i] = np.linalg.det(clusterVariance[i])
    if (finalDeter[i] == 0):
        clusterVariance[i] = clusterVariance[i] + varianceNoise
        finalDeter[i] = np.linalg.det(clusterVariance[i])
    finalInv[i] = np.linalg.inv(clusterVariance[i])

finalProbs = {}
for dish in dishes:
    prob = {}
    for i in range(k):
        prob[i] = probabilityModel(dishNp[dish[0]], clusterMean[i], finalDeter[i], finalInv[i])

    finalProbs[dish[0]] = prob

finalProbCX = {}
for cluster in range(k):
    finalProbCXInner = {}
    for dish in dishes:
        probSum = 0
        for i in range(k):
            probSum += finalProbs[dish[0]][i] * (1.0/k)

        # Get probCXi 
        probCX = (finalProbs[dish[0]][cluster] * (1.0/k)) / (probSum)
        finalProbCXInner[dish[0]] = probCX
    finalProbCX[cluster] = finalProbCXInner

def probability(dishNumber, cluster):
    return finalProbCX[cluster][dishNumber]



#Test Against 3 Dishes
total = 0
for cluster in range(k):
    print("------------------------------")
    print("Dish %d | Cluster %d: %f" %(10, cluster, probability(10, cluster)))
    print("Dish %d | Cluster %d: %f" %(450, cluster, probability(450, cluster)))
    print("Dish %d | Cluster %d: %f" %(600, cluster, probability(600, cluster)))
    print("Dish %d | Cluster %d: %f" %(2, cluster, probability(2, cluster)))





#Now Use This Model to Estimate Rating
user = "399"

def predict(userT, dishT):
    topSum = 0
    bottomSum = 0

    # Calculate What the User Thinks of Each Cluster

    # Itialize Sum to 0
    clusterRate = {}
    for i in range(k):
        clusterRate[i] = 0

    # Go Through All Previously Rated Dishes
    for dish in trainDict[user]:
        #print("Dish %d: %f" %(dish, trainDict[user][dish]))

        # Go Through Each Cluster adding What the User Usually Rates in that cluster
        for i in range(k):
            clusterRate[i] += probability(dish, i) * trainDict[user][dish]
            print(clusterRate[i])

    # Normalize by Number of Dishes
    for i in range(k):
        clusterRate[i] = clusterRate[i] / len(trainDict[user])
        #print(clusterRate[i])

    result = 0
    for i in range(k):
        result += clusterRate[i] * probability(dishT, i, clusterMean, clusterVariance)

    print(len(trainDict[user]))

    return result

print(predict(user, 813))