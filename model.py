# model.py

import numpy as np
import pandas as pd 

import json

dishesPath = './data/dishes.csv'
trainPath = './data/user_ratings_train.json'
testPath = './data/user_ratings_test.json'

dishes = pd.read_csv(dishesPath)


dishes = dishes.to_numpy()

#Iterate Through each dish
for dish in dishes:
    # dish[0] is the dish id
    # dish[1] is the dish name
    # ...
    pass

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
print("MAE: ", mae)


"""

Task 2 - Recommend dishes to users


"""

# Returns the number of recommended results as a list
def recommend(user, number):
    recommendation = {}

    for dish in dishes:
        #If dish not previously rated, predict and add to list
        if dish[0] not in trainDict[user]:
            recommendation[dish[0]] = predict(user, dish[0])

    #Sort recommendations
    