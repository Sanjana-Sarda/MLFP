import sys
sys.path.append("../../../")
import numpy as np
import os
from maraboupy import Marabou
from tensorflow import keras
import time
from scipy.special import softmax
import matplotlib.pyplot as plt

options = Marabou.createOptions(verbosity = 0, milpTightening="lp", solveWithMILP=True) #solveWithMILP=False)#

def setup(network, image):
    
    epsilon = 0.005
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]
    lb = np.clip(image-epsilon, 0, 1)
    ub = np.clip(image+epsilon, 0, 1)
    for h in range(inputVars.shape[0]):
        for w in range(inputVars.shape[1]):
            network.setLowerBound(inputVars[h][w], lb[inputVars[h][w]])
            network.setUpperBound(inputVars[h][w], ub[inputVars[h][w]])
    
    vals, stats = network.solve(options = options)
    output = [vals[a] for a in range (784, 794)]
    #p =  (softmax(output))
    
    return output

if __name__ == "__main__":
    start = time.time()
    # Load the mnist test data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Test 1
    print("Fully Connected Network Example")
    filename = "modelpr.onnx"
    #filename = "modelp1.onnx"
    inputName = 'Placeholder:0'
    outputName = 'y_out:0'
    
    indexes = []
    vals_1 = []
    vals_2 = []
    vals_3 = []
    vals_4 = []
    vals_5 = []
    vals_10 =[]
    
    
    for index in range(1000):
        #if index<=837:
          #  continue
        network = Marabou.read_onnx(filename=filename, inputNames=["flatten_4_input"], outputName = "dense_10")
        
        image = x_train[index]
        targetLabel = y_train[index]
        f = network.evaluateWithoutMarabou([image])
        prediction = np.argmax(f)
        
        #print ("Index"+str(index))
        
        if(prediction != targetLabel):
            #print ("Classified Wrong")
            continue
        
        
        #p =  (softmax(f))
        thing = np.sort(f)[0]
        #image = x_test[index].flatten() / 255
        #pp = setup(network, image)
        
        indexes.append(index)
        #max_vals.append(np.max(p))
        #print(np.max(p))
        #sec_vals.append(np.partition(p.flatten(), -2)[-2])
        #print (np.partition(p.flatten(), -2)[-2])
        #min_vals.append(np.min(p))
        #thing = np.partition(p.flatten(), -2)
        vals_1.append(thing[0])
        #vals_2.append(thing[8]-thing[7])
        #vals_3.append(thing[7])
        #vals_4.append(thing[6])
        #vals_5.append(thing[5])
        #vals_10.append(thing[0])
        
        #max_pvals.append(np.max(pp))
        #sec_pvals.append(np.partition(pp.flatten(), -2)[-2])
        #min_pvals.append(np.min(pp))
    
    #plt.figure()
    #plt.plot(indexes, max_vals, label = "Original Image")
    #plt.plot(indexes, max_pvals, label = "Perturbed Image")
    #plt.xlabel('Image Index')
    #plt.ylabel('Correct Label Probability')
    #plt.title("Train Correct Label")
    #plt.legend()
    #plt.savefig('max_train.jpg')
    
    #plt.figure()
    #plt.plot(indexes, sec_vals, label = "Original Image")
    #plt.plot(indexes, sec_pvals, label = "Perturbed Image")
    #plt.xlabel('Image Index')
    #plt.ylabel('Second Likely Label Probability')
    #plt.title("Train Second Likely Label")
    #plt.legend()
    #plt.savefig('sec_train.jpg')
    
    plt.figure()
    plt.scatter(indexes, vals_1, label = "1")
    #plt.scatter(indexes, vals_2, label = "2")
    #plt.scatter(indexes, vals_3, label = "3")
    #plt.scatter(indexes, vals_4, label = "4")
    #plt.scatter(indexes, vals_5, label = "5")
    #plt.scatter(indexes, vals_10, label = "10")
    plt.xlabel('Image Index')
    plt.ylabel('Values')
    plt.title("Train P1")
    plt.legend()
    plt.savefig('trainingp1.jpg') 
    
    end = time.time()
    print (end-start)
    exit(0)



