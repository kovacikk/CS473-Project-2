"""
model2.py

Kyle Kovacik
Model-Based Clustering Approach
- Predicts Ratings and Generates Recommendations
- Guassian Mixture Model

"""

import numpy as np
import pandas as pd
import random
import math

import json
import copy

random.seed(a= "slugcat23", version=2)

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
    - Soft Clustering
    - Gaussian Mixture Model


"""


"""

Task 1 - Predict User Rating From 1-5

- Using Mean Absolute Error (MAE) to evaluate

"""


# Set Number of Clusters
k = 5

# Number of Ingredients (or features)
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
    value = random.randint(0, len(dishes)-1)
    tester = dishNp[value]
    clusterMean.append(tester)
    
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

# Begin Iterating until Convergence (Or 20 Iterations)
for iteration in range(20):
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


    # Every 1 Iteration(s), Check if models have converged
    if (iteration % 1) == 0:

        #Check if all models have converged
        count = 0
        for kcluster in range(k):
            if ((np.linalg.norm(prevClusters[kcluster] - clusterMean[kcluster])) < 0.005):
                count = count + 1
        
        # All models are less than 0.005
        if (count == k):
            break


# Build Final Dish Probability Dictionary for Quick Probability

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



# Use Vector Space Simularity Between The User Rated Dishes and Cluster Vectors
def predict(userT, dishT):
    # Generate Vectors that represent how close to clusters dishes are
    vectorA = []

    for i in range(k):
        vectorA.append(probability(dishT, i))
    
    vectorA = np.array(vectorA)

    # Compute Cosine Simularity between dishes cluster vectors
    sim = {}
    for dish in trainDict[userT]:
        vectorB = []
        for i in range(k):
            vectorB.append(probability(dishT, i))
        vectorB = np.array(vectorB)

        #Cosine Simularity
        consineValue = np.dot(vectorA, vectorB) / (np.linalg.norm(vectorA) * np.linalg.norm(vectorB))
        sim[dish] = consineValue

    topSum = 0
    bottomSum = 0
    for dish in sim:
        topSum += sim[dish] * (trainDict[userT][dish] - averageRatings[userT])
        bottomSum += abs(sim[dish])

    if (bottomSum == 0):
        bottomSum = 1

    prediction = averageRatings[userT] + (topSum / bottomSum)
    return prediction



# Calculate MAE
mae = 0
n = 0
for user in test:
    for dishRating in test[user]:
        prediction = predict(user, dishRating[0])
        mae += abs(dishRating[1] - prediction)
        n += 1

mae = mae / n
print("Task 1 MAE: %f" %(mae))



"""

Task 2 - Recommend Dishes


"""

# Returns the number of recommended results as a list
def recommend(user, number):
    recommendation = {}

    for dish in dishes:
        #If dish not previously rated, predict and add to list
        if dish[0] not in trainDict[user]:
            recommendation[dish[0]] = predict(user, dish[0])

    #Sort recommendations
    recommendation = sorted(recommendation.items(), key = lambda r: (r[1], r[0]), reverse = True)
    return recommendation[:number]

rate = recommend("0", 10)


# Calculate Precision

averagePrecision10 = 0
averageRecall10 = 0

averagePrecision20 = 0
averageRecall20 = 0

userCount = len(test)
for user in test:

    totalRelevant = 0
    for dishRating in test[user]:
        # Dishes that are liked
        if (dishRating[1] >= 3):
            totalRelevant += 1

    rate20 = recommend(user, 20)

    #Find Relevant in 10 and 20
    relevant10 = 0
    relevant20 = 0
    for i in range(20):
        if rate20[i][1] >= 3:
            if (i < 10):
                relevant10 += 1
            relevant20 += 1

    precision10 = relevant10 / 10
    recall10 = relevant10 / totalRelevant

    precision20 = relevant20 / 20
    recall20 = relevant20 / totalRelevant


    averagePrecision10 += precision10
    averageRecall10 += recall10

    averagePrecision20 += precision20
    averageRecall20 += recall20

averagePrecision10 = averagePrecision10 / userCount
averageRecall10 = averageRecall10 / userCount

averagePrecision20 = averagePrecision20 / userCount
averageRecall20 = averageRecall20 / userCount


#Results

print("Task 2 Precision@10: %f" %(averagePrecision10))
print("Task 2 Precision@20: %f" %(averagePrecision20))

print("Task 2 Recall@10: %f" %(averageRecall10))
print("Task 2 Recall@20: %f" %(averageRecall20))