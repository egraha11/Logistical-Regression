import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
import random



class Logistical_Regression:

    bo = 0
    b = 0

    def sigmoid(self, bo, b, input):

        return(1/(1 + np.exp(-(bo + (b * input)))))


    def maximum_likelihood(self, train):
        
        #initialize intercept and coefficient parameters
        bo_min = b_min = np.min(train[0])
        bo_max = b_max = np.max(train[0])

        #initialize likelihood value and the intercept and coeffecient to be used in the model
        maximum_likelihood = 0
        bo = bo_min
        b = b_min

        #container for the cumulative likelihood
        likelihood = 0


        for i in np.arange(-15, 15, .5):
            for j in np.arange(-15, 15, .5):
                likelihood = 0
                for k in range(len(train[0])):
        
                
                    p = self.sigmoid(i, j, train[0][k])
                    likelihood += (p * train[1][k]) + ((1 - p) * (1 - train[1][k]))

                if likelihood > maximum_likelihood:
                    maximum_likelihood = likelihood
                    bo = i
                    b = j 

        self.bo = bo
        self.b = b

        x = np.arange(np.min(train[0]), np.max(train[0]), .1)
        y=[]
        for i in range(len(x)):
            y.append(self.sigmoid(self.bo, self.b, i))

        plt.plot(x, y)
        plt.scatter(train[0], train[1])
        plt.show()
        print("intercept: " + str(bo) + " " + "coefficient: " + str(b))

    def test(self, test):

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0 

        for i in range(len(test[1])):
            p = self.sigmoid(self.bo, self.b, test[0][i])
            
            if test[1][i] == 1:
                if p > .5:
                    true_positive += 1
                else:
                    false_positive += 1
            elif test[1][i] == 0:
                if p < .5:
                    true_negative += 1
                else:
                    false_negative += 1
        print("Test\ntrue positive: " + str(true_positive) + "\ntrue negative: " + 
        str(true_negative) + "\nfalse positive: " + str(false_positive) + "\nfalse negative: " + str(false_negative))



    def __init__(self, data1, data2):

        #split up data into testing and training arrays
        benign_train = np.stack((data1[0][0:round((len(data1[0]) * .8))], data1[1][0:round((len(data1[0]) * .8))]))
        malignant_train = np.stack((data2[0][0:round((len(data1[0]) * .8))], data2[1][0:round((len(data1[0]) * .8))]))

        train = np.concatenate((benign_train, malignant_train), axis=1)

        benign_test = np.stack((data1[0][(len(benign_train[0])):], data1[1][len(benign_train[1]):]))
        malignant_test = np.stack((data2[0][(len(malignant_train[0])):], data2[1][len(malignant_train[1]):]))

        test = np.concatenate((benign_test, malignant_test), axis=1)

        self.maximum_likelihood(train)

        self.test(test)


    def predict(self, independant_variable):

        p = self.sigmoid(self.bo, self.b, float(independant_variable))

        if p > .5:
            print("malignant with a probability of " + str(p))
        if p < .5:
            print("benign with a probability of " + str(1 - p))


#create a random dataset for testing
benign_radius = []
malignant_radius = []

for i in range(100):
    benign_radius.append(random.uniform(1, 15))
    malignant_radius.append(random.uniform(10, 25))

#assign a binary outcome to the independant variables
benign_results = np.full((1, 100), 0)
malignant_results = np.full((1, 100), 1)

#join the two arrays
benign = np.stack((benign_radius, benign_results[0]))
malignant = np.stack((malignant_radius, malignant_results[0]))


model = Logistical_Regression(benign, malignant)

while(True):
    val = input("Enter a radius size for the tumor in question or press ctrl C to quit: ")

    model.predict(val)
