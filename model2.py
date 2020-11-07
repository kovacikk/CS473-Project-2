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
#Iterate Through each dish
for dish in dishes:
    # dish[0] is the dish id
    # dish[1] is the dish name
    # ...

    temp = np.zeros(len(dish[2:]))
    for i in range(len(dish[2:])):
        temp[i] = dish[i + 2]
        #print(dish[i+2])
    dishNp[dish[0]] = np.array([temp])
    #print(dish[0])

#print(dishNp[200])

#print(dishNp[0][0][:10])
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

k = 20

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
    value = random.randint(0, len(dishes))
    test = dishNp[value]
    clusterMean.append(test)
    #print(value)
    #print(dishNp[value])
    #print(test)
    #print("------------------------------------")
    clusterVariance.append(np.array(temp2))



varianceNoise = clusterVariance[0]

#print(clusterVariance[0])
#print(clusterMean[0])

#print(dishes[0][2:])


def probability (x, mean, deter, inv):
    determinant = deter
    #determinant = np.linalg.det(covariance)

    #if (determinant == 0):
    #    covariance += varianceNoise
    #    determinant = np.linalg.det(covariance)

    #print(covariance, "\n----------\n", determinant)
    

    norm_const = 1.0 / math.sqrt(2 * np.pi * determinant)


    error = x - mean
    #print(np.matmul(x.T, x))
    #print(np.matmul(error, np.linalg.inv(covariance)))
    
    result = math.pow(math.e, -0.5 * np.linalg.multi_dot([error, inv, error.T]))

    return norm_const * result


#print(probability(dishes[0][2:], clusterMean[2], clusterVariance[2]))

# For each dish calculate the probability that it belongs in each cluster

# Run a Sum for the new mean and variance of a cluster

prevClusters = clusterMean

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
    for dish in dishes:
        prob = {}
        for i in range(k):
            prob[i] = probability(dishNp[dish[0]], clusterMean[i], deter[i], inv[i])

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
            probCX = (probs[dish[0]][cluster] * (1.0/k)) / probSum

            
            probStuff = (probCX / ( (1.0/k) * len(dishes) ))

            mean = np.add(mean, probStuff * dishNp[dish[0]])

            error = dishNp[dish[0]] - mean
            #print("Error:\n", error)
        
            #print(np.matmul(error.T, error))
            #print("\n--------------\n", np.matmul(error.T, error))
            #exit()
            variance = np.add(variance, probStuff * np.matmul(error.T, error))
            
            #if (dish[0] == 1):
                #exit()

        clusterMean[cluster] = mean
        clusterVariance[cluster] = variance

        

        #This is fucked for some reason

    #print(clusterVariance[0])
    #For Each Iteration
    if (iteration % 5) == 0:
        print("-------------------------------------")
        print("Iteration: %d" %(iteration))
        for kcluster in range(k):
            print("Cluster %d: %f" % (kcluster, np.linalg.norm(prevprevClusters[kcluster] - clusterMean[kcluster])))
            
        #print("5: ", np.linalg.norm(lastCluster0 - clusterMean[5]))
        #print(lastCluster0[0][:10])
        #print(clusterMean[0][0][:10])
        #print(clusterVariance[0][0][:10])
        
        #Check if all models have converged
        count = 0
        for kcluster in range(k):
            if ((np.linalg.norm(prevprevClusters[kcluster] - clusterMean[kcluster])) < 0.005):
                count = count + 1
        #print(count)
        
        # All models are less than 0.005
        if (count == k):
            break

        

        #exit()
    #if (np.array_equal(lastCluster0, clusterMean[0])):
    #    print("true")
    #    exit()


    # Update mean and variance for guassians based on dish probability 


    #Repeat Above until convergence is found

#print
#print(clusterVariance[0])




# Test Probabilities for Clusters

probs = {}
#To Save Time Calculate All Probabilities

#print("Dish:", dishNp[dishes[0][0]])

inv = {}
deter = {}
for i in range(k):
    deter[i] = np.linalg.det(clusterVariance[i])
    if (deter[i] == 0):
        clusterVariance[i] = clusterVariance[i] + varianceNoise
        deter[i] = np.linalg.det(clusterVariance[i])
    inv[i] = np.linalg.inv(clusterVariance[i])


prob = {}

for i in range(k):
    prob[i] = probability(dishNp[dishes[0][0]], clusterMean[i], deter[i], inv[i])

probs[dishes[0][0]] = prob

for i in range(k):

    probSum = 0
    for j in range(k):
        probSum += probs[dishes[0][0]][j] * (1.0/k)


    # Get probCXi 
    probCX = (probs[dishes[0][0]][i] * (1.0/k)) / probSum

    print("Cluster % d: % f" %(i, probCX))
    #print(clusterMean[i])







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
        prob[i] = probability(dishNp[dish[0]], clusterMean[i], deter[i], inv[i])

    probs[dish[0]] = prob


for dish in dishes[:5]:
    print("---------------------")
    probSum = 0
    for j in range(k):
        probSum += probs[dish[0]][j] * (1.0/k)

    for i in range(k):
        # Get probCXi 
        probCX = (probs[dish[0]][i] * (1.0/k)) / probSum

        print("Dish: %d | Cluster % d: % f" %(dish[0], i, probCX))
        #print(clusterMean[i])

