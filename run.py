import numpy as np
import pandas as pd

# read file into panda df
def pandas_reader(filename):
    df = pd.read_csv(filename)
    df = df.set_index('id')
    return df
def convert_numpy(train):
    # x be all the words
    train_x = train[list(train.keys())[:-1]]
    # let y be the label of trainging set
    train_y = train['label']
    np_train_x = train_x.to_numpy() 
    np_train_y = train_y.to_numpy()
    return np_train_x, np_train_y

    return np_train_x, np_train_y
def export_csv(test_noans, out_label):
    # put labels into test with noans first
    test_noans["label"] = out_label

    # get id and label col
    test_answer = test_noans["label"]

    # export to csv
    test_answer.to_csv("test_answer.csv")

# this function will calculate p using dot product
# p = 1 / (1 + e^ -(x*w))
def getp(x, w):
    z = np.dot(x, w)
    p = 1 / (1 + np.exp(-z))
    return p
# returns λ||w||2
def getNorm(lamb, w):
    norm = 0
    for el in w:
        norm += el**2
    norm = np.sqrt(norm)
    norm = lamb* norm
    return norm
# w is the initilized weight vector
# a is the learning rate, lamb is lambda.
# we should output the converged weight vector
def SGD(x, y, w, a, lamb):
    
    # calculte LCL based on the current weight vector
    # LCL = sum of (yi*log p + (1-yi)* log(1-p))
    # for every i, yi is the label, and p should be different
    LCL = 0
    new = 0
    prev = 0
    for i in range(len(x)):
        p = getp(x[i], w) # the p for this data
        LCL += (y[i]*np.log(p) + (1-y[i]) * np.log(1-p)) # sum up the LCL
    
    norm = getNorm(lamb, w)
    new = LCL - norm
    
    # stop the while loop when further iterations won't change the weight anymore
    # checking if the change in LCL − λ||w||2 is less than a small value 1.0 × 10−3 or 1.0 × 10−4
    # ||w||2 = sqrt(w1^2 + w2^2 + w3^2 + ... wn^2)
    #  while "objective function do not converge":
    epoch = 0

    while np.abs(new - prev) > 10 ** -3 :
        print(np.abs(new - prev), p, y[-1])
        epoch += 1
        
        print("epoch:", epoch)
#         if epoch > 150:
#             break
            
        # shuffle the training data
        # Get a random permutation of indices
        indices = np.random.permutation(len(x))

        # Shuffle x and y using the same permutation
        x = x[indices]
        y = y[indices]
        
        for i in range(len(x)):
            p = getp(x[i], w) # current p value
            # w = w + α((y − p)x − 2λw );
            res = [element * 2 * lamb for element in w] # 2 *lamb * w

            w = w + a*((y[i] - p)*x[i] - res)
        
        prev = new # record the previous LCL - norm
        LCL = 0 # recalculate new LCL here
        for i in range(len(x)):
            p = getp(x[i], w) # the p for this data
            LCL += (y[i]*np.log(p) + (1-y[i]) * np.log(1-p)) # sum up the LCL
        norm = getNorm(lamb, w)
        new = LCL - norm
        # dacay learning rate
        a = 0.975 * a

    # after the while loop ends, we return the optimized weight

    return w

def predict(w, test_x):
    # compute the labels using the weight vectors we have
    output = []
    for j in range(len(test_x)):
        p = getp(test_x[j], w)
        p1 = p # probability of label = 1
        p0 = 1 - p # probability of label = 0
        if p1 > p0 :
            output.append(1)
            
        else:
            output.append(0)
    return output

"""
x is the features vector
w is the weight vector
p = 1 / (1 + e^ (- x*w))

P(Y=1 | x,w) = log p
P(Y=0 | x,w) = log(1-p)


gradient for p = p(1-p)x^j

gradient log P(Y=1 | x,w) = (1-p)x^j
gradient log P(Y=0 | x,w) = -px^j

gradient log P(Y=y | x,w) = (y-p)x^j

increment x:
alpha is the learning rate
w^(t+1) = w^t + alpha (y-p)x


with L2, update:
w^(t+1) = w^t + alpha ((y-p)x - 2 lambda w^t)

"""


def main():
    train = pandas_reader("train.csv")
    x, y = convert_numpy(train)
    test = pandas_reader("test_noans.csv")
    test_x = test.to_numpy() 

    # the variables used for running the model
    w = [0] * len(x[0])
    a = 0.1 # our choice of learning rate
    lamb = 0.00001 # our choice of lambda

    weight = SGD(x, y, w, a, lamb)

    output = predict(weight, test_x)
    export_csv(test, output)




if __name__ == "__main__":
    main()