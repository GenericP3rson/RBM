import numpy as np
import pandas as pd 
import torch 
# import torch.nn as nn
# from torch.autograd import Variable

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# Just to do stuff with the weird input file

users = pd.read_csv('ml-1m/users.dat', sep='::', header=None,
                    engine='python', encoding='latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat', sep='::',
                      header=None, engine='python', encoding='latin-1')

# print(movies, users, ratings)

trainingSet = np.array(pd.read_csv('ml-100k/u1.base', delimiter = '\t'), dtype='int')
print(trainingSet)
testSet = np.array(pd.read_csv('ml-100k/u1.test', delimiter = '\t'), dtype='int')
print(testSet)

totalUsers = int(max(max(trainingSet[:,0]), max(testSet[:, 0])))
# Num of users; max movie id
totalMovies = int(max(max(trainingSet[:, 1]), max(testSet[:, 1])))

print(totalUsers, totalMovies)

# Array with user i by movie x
# Observations in lines, features in columns
# data = []
# for i in range((totalUsers)):
#     user = []
#     for j in range((totalMovies)):
#         seen = False
#         for q in trainingSet:
#             if (q[0] == i and q[1] == j):
#                 user.append(q)
#                 seen = True
#                 break 
#         if (not seen):
#             user.append(None)
#         print(user)
#     data.append(user)
#     user = []
#     print(data)
# # print(np.array(data))

def parse(arr):
    # Basically makes an array of users x movies
    data = []
    for i in range(1, totalUsers+1):
        movies = arr[:, 1][arr[:, 0] == i]
        # All movies ids if id == 1 or whatever the use number is
        ratings = arr[:, 2][arr[:, 0] == i]
        usrRate = np.zeros(totalMovies)
        usrRate[movies - 1] = ratings
        # For zero-indexing...
        data.append(list(usrRate))
    return data


# print(parse(trainingSet))
# print(parse(testSet))

# Tensors == array of a single data type
# Like a PyTorch array...
# Same thing with TF

trainingSet = torch.FloatTensor(parse(trainingSet))
testSet = torch.FloatTensor(parse(testSet))


# Now we actually do it!
trainingSet[trainingSet == 0] = -1
trainingSet[trainingSet == 2] = 0
trainingSet[trainingSet == 1] = 0
trainingSet[trainingSet >= 3] = 1
testSet[testSet == 0] = -1
testSet[testSet == 2] = 0
testSet[testSet == 1] = 0
testSet[testSet >= 3] = 1

print(trainingSet, trainingSet)

class RBM():
    def __init__(self, visibleNodes, hiddenNodes):
        '''
        Input: Num of total movies, num of users
        '''
        self.vNodes = visibleNodes
        self.hNodes = hiddenNodes
        self.weights = torch.randn(hiddenNodes, visibleNodes)
        self.biasesHiddenNodes = torch.randn(1, hiddenNodes)
        self.biasesVisibleNodes = torch.randn(1, visibleNodes)

    def sampleHiddenNodes(self, x):
        wx = torch.mm(x, self.weights.t())
        act = wx + self.biasesHiddenNodes.expand_as(wx)
        # Applies to each batch
        '''
        Activation function is the probability!
        '''
        probHiddenActGivenVisible = torch.sigmoid(act)
        return probHiddenActGivenVisible, torch.bernoulli(probHiddenActGivenVisible)

    def sampleVisibleNodes(self, y):
        wy = torch.mm(y, self.weights)
        act = wy + self.biasesVisibleNodes.expand_as(wy)
        # Applies to each batch
        '''
        Activation function is the probability!
        '''
        probVisibleActGivenHidden = torch.sigmoid(act)
        return probVisibleActGivenHidden, torch.bernoulli(probVisibleActGivenHidden)
    
    def train(self, inputVector, visibleNodesAfterKSamplings, probInputVector, probVisibleNodes):
        self.weights += (torch.mm(inputVector.t(), probInputVector) - torch.mm(visibleNodesAfterKSamplings.t(), probVisibleNodes)).t()
        self.biasesVisibleNodes += torch.sum((inputVector - visibleNodesAfterKSamplings), 0) # Keeps format of b as two dimensions
        self.biasesHiddenNodes += torch.sum((probInputVector - probVisibleNodes), 0)

numVisibleNodes = len(trainingSet[0])
numHiddenNodes = 100 # We can tune it: it's how many features
batchSZ = 100

i = RBM(numVisibleNodes, numHiddenNodes)

epochs = 10
for x in range(epochs):
    loss = 0
    # Most common: RMSE (root mean squared error)
    count = 0.0
    for users in range(0, totalUsers - batchSZ, batchSZ): 
        # So we iterate between all the batches...
        visibleVector = trainingSet[users:users+batchSZ]
        originalNodes = trainingSet[users:users+batchSZ] # Movies already rated
        initProb,_ = i.sampleHiddenNodes(originalNodes)
        # Probabilities of hidden nodes
        for k in range(10): # Contrastive divergence for ksteps
            _,hiddenAtK = i.sampleHiddenNodes(visibleVector)
            _,visibleVector = i.sampleVisibleNodes((hiddenAtK))
            visibleVector[originalNodes<0] = originalNodes[originalNodes < 0]
        probHidden,_ = i.sampleHiddenNodes(visibleVector)
        i.train(originalNodes, visibleVector, initProb, probHidden)
        # ,_ means only the first element
        loss += torch.mean(torch.abs(originalNodes[originalNodes>=0] - visibleVector[originalNodes>=0]))
        count += 1.0
    print(f"{x} epoch; {loss/count} normalised loss")

        
# i.sampleHiddenNodes()





'''
The testing phase...
'''
lossTest = 0
# Most common: RMSE (root mean squared error)
count = 0.0
for users in range(totalUsers):
    # So we iterate between all the batches...
    inp = trainingSet[users:users+1] # Has the unrated
    # Movies already rated
    target = testSet[users:users+1]
    # initProb, _ = i.sampleHiddenNodes(originalNodes)
    # Probabilities of hidden nodes
    # for k in range(10):  # Contrastive divergence for ksteps
    # We just make one step because we have already made the vector stuff!
    if (len(target[target>=0]) > 0):
        _, hiddenVector = i.sampleHiddenNodes(inp)
        _, visibleVector = i.sampleVisibleNodes((hiddenVector))
        # visibleVector[originalNodes < 0] = originalNodes[originalNodes < 0]
        # Just one round of Gibbs Sampling.
        # probHidden, _ = i.sampleHiddenNodes(visibleVector)
        # i.train(originalNodes, visibleVector, initProb, probHidden)
        # ,_ means only the first element
        lossTest += torch.mean(torch.abs(
            target[target >= 0] - visibleVector[target >= 0]))
        count += 1.0
print(f"{lossTest/count} normalised test loss")

