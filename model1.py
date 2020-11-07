# model.py

import numpy as np
import pandas as pd 

import json

dishesPath = './data/dishes.csv'
trainPath = './data/user_ratings_train.json'
testPath = './data/user_ratings_test.json'

dishes = pd.read_csv(dishesPath)

dishes = dishes.to_numpy()


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
    Memory Based Collaborative Filtering
    - Using Vector Space Similarity
    - Given a user u, find similar users
        > Predict u's ratings based on similar user ratings


"""




"""

Task 1 - Predict User Rating From 1-5

- Using Mean Absolute Error (MAE) to evaluate

"""

sim = {}

# Find Vector Space Simularity for All Users
for userA in train:
    innerSim = {}
    for userB in train:
        
        #Calculate Cosine Simularity
        cosineValue = np.dot(ratingVectors[userA], ratingVectors[userB]) / (np.linalg.norm(ratingVectors[userA]) * np.linalg.norm(ratingVectors[userB]))

        innerSim[userB] = cosineValue
    
    sim[userA] = innerSim
    


def predict(userT, dish):
    topSum = 0
    bottomSum = 0
    for user in sim:
        #Only Sum on users that rated the dish we are predicting
        if dish in trainDict[user]:
            topSum += sim[user][userT] * (trainDict[user][dish] - averageRatings[user])
            bottomSum += abs(sim[user][userT])

    # Avoid Divide by Zero
    if bottomSum == 0:
        bottomSum = 1
 
    prediction = averageRatings[userT] + (topSum / bottomSum)
    return prediction


# Evaluation:
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

Task 2 - Recommend dishes to users
    > Calculate Precision and Recall

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

#print("Precision10:", averagePrecision10)
#print("Recall10:", averageRecall10)

averagePrecision20 = averagePrecision20 / userCount
averageRecall20 = averageRecall20 / userCount

#print("Precision20:", averagePrecision20)
#print("Recall20:", averageRecall20)

print("Task 2 Precision@10: %f" %(averagePrecision10))
print("Task 2 Precision@20: %f" %(averagePrecision20))

print("Task 2 Recall@10: %f" %(averageRecall10))
print("Task 2 Recall@20: %f" %(averageRecall20))