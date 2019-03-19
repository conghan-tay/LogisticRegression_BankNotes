import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

def LoadShuffledData():
    X_shuffled = dataLoad("a4shuffled.data")
    X_norm_shuffled = dataNorm(X_shuffled)
    return X_shuffled, X_norm_shuffled

def loadInitialData():
    X = dataLoad("data_banknote_authentication.txt")
    X_norm = dataNorm(X)
    return X, X_norm

def dataLoad(filename):
  # load data from filename into X
  X=[]
  
  text_file = open(filename, "r")
  lines = text_file.readlines()
    
  for line in lines:
    words = line.split(",")
    # convert value of first attribute into float
    Row = []
    for word in words:
        Row.append(float(word))
    X.append(Row)
    
  return np.asarray(X)

def dataNorm(X):
    # Last Col of X is the target output
    # so ignore that
    X_min = np.min(X[:, 0:X.shape[1]-1], axis = 0)
    X_max = np.max(X[:, 0:X.shape[1]-1], axis = 0)
    X_norm = (X[:, 0:X.shape[1]-1] - X_min)/ (X_max - X_min)
    X0 = np.ones((X_norm.shape[0],1))
    
    # for each col/feature
    for i in range(X_norm.shape[1]):
        X_app = X_norm[:,i]
        X_app = np.reshape(X_app, (X_norm.shape[0], 1))
        
        # append the feature to X0 as a column
        X0 = np.append(X0, X_app, axis = 1)
    
    # final col is target output
    # get target output and append it as col
    # to X0
    X_app = X[:,X.shape[1] - 1]
    X_app = np.reshape(X_app, (X.shape[0], 1))
    X0 = np.append(X0,X_app, axis = 1)
    return X0


def splitCV(X_norm, K):
    X_split = []
    m = X_norm.shape[0]
    each_set = m//K
    np.random.shuffle(X_norm)
    '''
    for i in range(k-1):
        curr_fold = X_norm[i*each_set:(i+1)*each_set,:]
        
    '''
    for i in range(K - 1):
        curr_fold = X_norm[i*each_set:(i+1)*each_set,:]
        X_split.append(curr_fold)
    X_split.append(X_norm[(K - 1)*each_set:,:])
    
    return X_split

'''

        print("KFolds")
        print(Y_hat.shape)
        print("\n")
        print(Y.shape)
        print("\n")
        
        print("RMSE for " + str(i) + " is ")
        print(Error)
        print("\n")
        
'''

def SaveSplit(filename, X_split):
    with open(filename, 'w') as outfile:
        for i in range(len(X_split)):
            for j in range(X_split[i].shape[0]):
                for k in range(X_split[i][j].shape[0]):
                    outfile.write(str(X_split[i][j][k]))
                    outfile.write(' ')
                outfile.write('\n')
                '''
        for i in X_split:
            for j in i:
                for k in i:
                    #print(k)
                    #print('\n')
                    #np.savetxt(outfile, k, delimiter=" ", fmt='%10.5f')
                    for z in k:
                        outfile.write(str(z))
                        outfile.write(' ')
                    outfile.write('\n')
                #outfile.write('\n')'''
                    

def gradientDescentKFolds(X, Theta, Alpha, Num_iters, K, savename):
    X_split = splitCV(X, K)
    saveSplitName = savename + "Data.txt"
    SaveSplit(saveSplitName,X_split)
    '''
    plt.clf()
    plt.title(str(K) +  " Folds")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    '''
    
    #Iter_list = range(0, K)
    ErrorList = []
    Predicted = None
    Actual = None    
    
    for i in range(K):
        X_train, X_test = trainTestSplit(i, X_split)
        C_Theta = stochasticGD(X_train, Theta, Alpha, Num_iters)
        
        Y_hat = np.matmul(X_test[:,0:X_test.shape[1] - 1], C_Theta)
        Y_hat = np.reshape(Y_hat,(X_test.shape[0],1))

        if Predicted is None:
            Predicted = Y_hat
        else:
            Predicted = np.vstack([Predicted,Y_hat])
        
        Y = X_test[:,X_test.shape[1] - 1]
        Y = np.reshape(Y, (X_test.shape[0],1))

        if Actual is None:
            Actual = Y
        else:
            Actual = np.vstack([Actual,Y])        
        
        ErrorList.append(rmse(Y_hat, Y))
        
    
    '''
    plt.plot(Iter_list,ErrorList)  
    plt.savefig(savename)
    '''
    print(ErrorList)
    
    EL_np = np.array(ErrorList)
    
    savename= savename + '.csv'
    EL_np.tofile(savename,sep="\n",format='%10.5f')
    
    savename = savename + "Predicted.csv"
    Predicted.tofile(savename,sep="\n",format='%10.5f')
    
    savename = savename + "Actual.csv"
    Actual.tofile(savename,sep="\n",format='%10.5f')


'''
Takes in index of test and Splitted X
Returns Combined Train and Seperated Test with index
'''
def trainTestSplit(index, X_split):
    test = X_split[index]
    rest = np.delete(X_split, index,0)
    train = []
    for sublist in rest:
        for item in sublist:
            train.append(item)
    
    return np.array(train), np.array(test)

def PlotAllOutputInput(X):
    for i in range(X.shape[1]-1):
        inputOutputPlot("Attr"+str(i)+".png", str(i), X[:,i], X[:,X.shape[1]-1])

def inputOutputPlot(filename, attr, inputattr, outputattr):
    plt.clf()
    plt.title("Input Vs Output")
    plt.xlabel("attr" + attr)
    plt.ylabel("Output")
    plt.plot(inputattr,outputattr,'ro')
    plt.savefig(filename)

def gradientDescentTest(X_norm):
    plt.clf()
    plt.title("Cost")
    
    '''
    np.random.shuffle(X_norm)
    stochasticGD(X_norm, np.zeros((X_norm.shape[1]-1,1)),0.001,1500)
    np.random.shuffle(X_norm)
    stochasticGD(X_norm, np.zeros((X_norm.shape[1]-1,1)),0.003,1500)
    np.random.shuffle(X_norm)
    stochasticGD(X_norm, np.zeros((X_norm.shape[1]-1,1)),0.01,1500)
    np.random.shuffle(X_norm)
    stochasticGD(X_norm, np.zeros((X_norm.shape[1]-1,1)),0.03,1500)
    np.random.shuffle(X_norm)
    stochasticGD(X_norm, np.zeros((X_norm.shape[1]-1,1)),0.1,1500)
    np.random.shuffle(X_norm)
    stochasticGD(X_norm, np.zeros((X_norm.shape[1]-1,1)),0.3,1500)
    np.random.shuffle(X_norm)
    stochasticGD(X_norm, np.zeros((X_norm.shape[1]-1,1)),0.6,1500)
    np.random.shuffle(X_norm)
    '''
    stochasticGD(X_norm, np.zeros((X_norm.shape[1]-1,1)),1.0,1500)
    #gradientDescent(X_norm, np.zeros((14,1)),1,1500)
   
    filename = "GradientDescent.png"
    plt.legend(['1.0'])
    #plt.legend(['0.001', '0.003','0.01', '0.03', '0.1', '0.3', '0.6', '1.0'])
    plt.savefig(filename, format = "png") 
    
# np.random.rand(14,1), 0.001, 4500
def stochasticGD(X, Theta, Alpha, Num_iters):
    No_Y = X[:, :X.shape[1] - 1]
    No_Y = np.reshape(No_Y, (X.shape[0], X.shape[1] - 1))
    Y =  X[:, X.shape[1] - 1]
    Y = np.reshape(Y, (X.shape[0],1))
    New_Theta = Theta
    
    Iter_list = range(0, Num_iters)
    Error = []
    m = X.shape[0]
    
    
    for i in range(Num_iters):
        j = i % m
        X_j = No_Y[j,:]
        X_j = np.reshape(X_j,(1,No_Y.shape[1]))
        Y_hat_j = np.matmul(X_j,New_Theta)
        Y_hat_j = ss.expit(Y_hat_j)
        Y_j = Y[j,:]
        Y_j = np.reshape(Y_j, (1,1))
        E = Y_hat_j  - Y_j #E is (1,1)
        delta =E * X_j
        Theta_Change = Alpha * delta
        New_Theta = New_Theta - Theta_Change.T
        Error.append(float(errCompute(X, New_Theta)))
        
    
    plt.plot(Iter_list, Error)
    #filename = "GradientDescent" +str(Alpha) + " " + str(Num_iters)+".png"
    #plt.savefig(filename, format = "png") 
    
    #print(New_Theta)
    return New_Theta

def rmse(testY, stdY):
    return np.sqrt(np.mean((testY - stdY)**2))

def Predict(X, Theta):
    Y_hat = predict_for_err(X, Theta)
    return Y_hat > 0.5

def predict_for_err(X,Theta):
    No_Y = X[:, :X.shape[1] - 1]
    No_Y = np.reshape(No_Y, (X.shape[0], X.shape[1] - 1))
    Y =  X[:, X.shape[1] - 1]
    Y = np.reshape(Y, (X.shape[0],1))
    Y_hat = np.matmul(No_Y, Theta)
    Y_hat = ss.expit(Y_hat)
    return Y_hat

def predict2file(Y_hat, filename):
    with open(filename, 'w') as outfile:
        for i in range(Y_hat.shape[0]):
            outfile.write(str(int(Y_hat[i])))
            outfile.write('\n')

def errCompute(X, Theta):
    m = X.shape[0]
    Y_hat = predict_for_err(X,Theta)
    Y =  X[:, X.shape[1] - 1]
    Y = np.reshape(Y, (X.shape[0],1))
    J = (-1.0/(m)) * np.sum((Y*np.log(Y_hat)) + ((1.0-Y)*np.log(1.0-Y_hat)))
    return J



def SaveTheta(filename, Theta):
    with open(filename, 'w') as outfile:
        for i in range(len(Theta)):
            for j in range(Theta[i].shape[0]):
                outfile.write(str(Theta[i][j][0]))
                outfile.write(",")
            outfile.write("\n")        


def Accuracy(X_norm,percent_train):
    X_test_01, X_test_02, X_test_03, X_test_04, X_test_05, All_Theta= splitTT(X_norm,percent_train)
    
    Y_hat = Predict(X_test_01, All_Theta[0])
    
    Y =  X_test_01[:, X_test_01.shape[1] - 1]
    
    Y = np.reshape(Y, (len(Y),1))
    Num_Wrong = np.sum(np.abs(Y_hat - Y))
    
    Percent_Acc = 1.0-(float(Num_Wrong)/float(X_test_01.shape[0]))
    print(Percent_Acc)
    
    Y_hat = Predict(X_test_02, All_Theta[1])
    Y =  X_test_02[:, X_test_02.shape[1] - 1]
    Y = np.reshape(Y, (len(Y),1))
    
    
    Num_Wrong = np.sum(np.abs(Y_hat - Y))
    Percent_Acc = 1.0-(float(Num_Wrong)/float(X_test_02.shape[0]))
    print(Percent_Acc)
    
    Y_hat = Predict(X_test_03, All_Theta[2])
    Y =  X_test_03[:, X_test_03.shape[1] - 1]
    Y = np.reshape(Y, (len(Y),1))
    
    Num_Wrong = np.sum(np.abs(Y_hat - Y))
    Percent_Acc = 1.0-(float(Num_Wrong)/float(X_test_03.shape[0]))
    print(Percent_Acc)
    
    Y_hat = Predict(X_test_04, All_Theta[3])
    Y =  X_test_04[:, X_test_04.shape[1] - 1]
    Y = np.reshape(Y, (len(Y),1))
    
    Num_Wrong = np.sum(np.abs(Y_hat - Y))
    Percent_Acc = 1.0-(float(Num_Wrong)/float(X_test_04.shape[0]))
    print(Percent_Acc)
    
    Y_hat = Predict(X_test_05, All_Theta[4])
    Y =  X_test_05[:, X_test_05.shape[1] - 1]
    Y = np.reshape(Y, (len(Y),1))
    
    Num_Wrong = np.sum(np.abs(Y_hat - Y))
    Percent_Acc = 1.0-(float(Num_Wrong)/float(X_test_05.shape[0]))
    print(Percent_Acc)
    

def splitTT(X_norm, percent_train):
    m = X_norm.shape[0]
    train_number = int(percent_train * m)
    np.random.seed(0)    
    
    Alpha = 1.0
    Num_iters = 1372 * 20
    All_Theta = []
    
    np.random.shuffle(X_norm)
    X_train_01 = X_norm[: train_number, :]
    X_test_01 = X_norm[train_number:, :]
    SaveSplit("X_split1.txt", [X_train_01, X_test_01])
    Theta = np.zeros((X_train_01.shape[1]-1,1))
    New_Theta = stochasticGD(X_train_01, Theta, Alpha, Num_iters)
    All_Theta.append(New_Theta)
    
    np.random.shuffle(X_norm)
    X_train_02 = X_norm[: train_number, :]
    X_test_02 = X_norm[train_number:, :]
    SaveSplit("X_split2.txt", [X_train_02, X_test_02])
    Theta = np.zeros((X_train_02.shape[1]-1,1))
    New_Theta = stochasticGD(X_train_02, Theta, Alpha, Num_iters)
    All_Theta.append(New_Theta)
    
    np.random.shuffle(X_norm)
    X_train_03 = X_norm[: train_number, :]
    X_test_03 = X_norm[train_number:, :]
    SaveSplit("X_split3.txt", [X_train_03, X_test_03])
    
    Theta = np.zeros((X_train_03.shape[1]-1,1))
    New_Theta = stochasticGD(X_train_03, Theta, Alpha, Num_iters)
    All_Theta.append(New_Theta)
    
    np.random.shuffle(X_norm)
    X_train_04 = X_norm[: train_number, :]
    X_test_04 = X_norm[train_number:, :]
    SaveSplit("X_split4.txt", [X_train_04, X_test_04])
    
    Theta = np.zeros((X_train_04.shape[1]-1,1))
    New_Theta = stochasticGD(X_train_04, Theta, Alpha, Num_iters)
    All_Theta.append(New_Theta)
    
    np.random.shuffle(X_norm)
    X_train_05 = X_norm[: train_number, :]
    X_test_05 = X_norm[train_number:, :]
    SaveSplit("X_split5.txt", [X_train_05, X_test_05])
    
    Theta = np.zeros((X_train_05.shape[1]-1,1))
    New_Theta = stochasticGD(X_train_05, Theta, Alpha, Num_iters)
    All_Theta.append(New_Theta)
    
    SaveTheta("Thetas.csv", All_Theta)
    
    return X_test_01, X_test_02, X_test_03, X_test_04, X_test_05, All_Theta
































    

  