# model.py

import numpy as np
import pandas as pd
import random
import math

import json

dishesPath = './data/dishes.csv'
trainPath = './data/user_ratings_train.json'
testPath = './data/user_ratings_test.json'

dishes = pd.read_csv(dishesPath)


dishes = dishes.to_numpy()

dishNp = {}
#Iterate Through each dish
for dish in dishes:
    # dish[0] is the dish id
    # dish[1] is the dish name
    # ...

    temp = np.zeros(len(dish[2:]))
    for i in range(len(dish[2:])):
        temp[i] = dish[i + 2]
    dishNp[dish[0]] = np.array([temp])


#print(len(dishes))


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

#Somehow cluster all the documents??????
# Soft K-Means


#Randomly Select k documents to be the initial cluster centers

# Gaussian Mixture Models with Expectation Maximization????



arr1 = np.array([[1, 0, 0, 4]])
arr2 = np.array([[5, 6],[7, 8]])

arr3 = np.array([dish[2:]])
test = np.array(dishes[0])

ar1 = np.zeros(2)
#print(np.matmul(ar1, ar1.T))

test = dishNp[0]
#print(np.matmul(test.T, test))
#exit()

#I don't know calculate L

k = 5

#Initialize Weights
weights = {}
for dish in dishes:
    weight = {}
    for i in range(k):
        weight[i] = 1 / k

    weights[dish[0]] = weight


#print(weights[0][1]) #Finds the weights for cluster 1 for dish 0

ingredients = len(dishes[1]) - 2


# Guess k gaussians - means and variance
clusterMean = []
clusterVariance = []
# Iterate Over Clusters
for i in range(k):
    
    # Iterate Over Ingredients
    temp2 = []
    for j in range(ingredients):
        # Generate Numbers between 0 and 1

        row = []
        for n in range(j):
            row.append(0)
        row.append(1)
        for n in range(ingredients - j - 1):
            row.append(0)
        temp2.append(row)

    clusterMean.append(dishNp[random.randint(0, len(dishes))])
    clusterVariance.append(np.array(temp2))


#print(clusterVariance[0])
#print(clusterMean[0])

#print(dishes[0][2:])


def probability (x, mean, covariance):
    determinant = np.linalg.det(covariance)

    norm_const = 1.0 / math.sqrt(2 * np.pi * determinant)


    error = x - mean
    #print(np.matmul(x.T, x))
    #print(np.matmul(error, np.linalg.inv(covariance)))

    result = math.pow(math.e, -0.5 * (np.matmul(np.matmul(error, np.linalg.inv(covariance)), error.T)))

    

    return norm_const * result 


#print(probability(dishes[0][2:], clusterMean[2], clusterVariance[2]))

# For each dish calculate the probability that it belongs in each cluster

# Run a Sum for the new mean and variance of a cluster

probs = {}
#To Save Time Calculate All Probabilities
for dish in dishes[:100]:
    prob = {}
    for i in range(k):
        prob[i] = probability(dishNp[dish[0]], clusterMean[i], clusterVariance[i])

    probs[dish[0]] = prob

#print(probs[dish][cluster])

#print(dishes[:100])

#Update Mean and Covariance for Each Cluster
for cluster in range(k):

    mean = np.zeros(ingredients)
    variance = np.zeros((ingredients, ingredients))

    for dish in dishes[:100]:

        #Get ProbXiC for Each Cluster
        probSum = 0
        for i in range(k):
            probSum += probs[dish[0]][i] * (1.0/k)


        # Get probCXi 
        probCX = (probs[dish[0]][cluster] * (1.0/k)) / probSum

        
        probStuff = (probCX / ( (1.0/k) * len(dishes) ))

        mean = np.add(mean, probStuff * dishNp[dish[0]])

        error = dishNp[dish[0]] - mean
        
       
        #print(np.matmul(error.T, error))
        

        variance = np.add(variance, probStuff * np.dot(error, error.T))
        #print(variance)
        #if (dish[0] == 1):
            #exit()

    clusterMean[cluster] = mean
    clusterVariance[cluster] = variance
    print(clusterVariance) #This is fucked for some reason


#print(clusterMean)


# Update mean and variance for guassians based on dish probability 


#Repeat Above until convergence is found












