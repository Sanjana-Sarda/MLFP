import sys
sys.path.append("../../../../")
import numpy as np
import os
from maraboupy import Marabou
from tensorflow import keras
import time
import copy
from PIL import Image

options = Marabou.createOptions(verbosity = 0, milpTightening="lp", solveWithMILP=True, timeoutInSeconds=60) #solveWithMILP=False)#


def top1(network, image, targetLabel, results, epsilon):
    
    #network.setLowerBound(epsilon, 0) #ub = 0.9 lb = 0 (0.5)
    #network.setUpperBound(epsilon, 1)
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]
    lb = np.clip(image-epsilon, 0, 1)
    ub = np.clip(image+epsilon, 0, 1)
    for h in range(inputVars.shape[0]):
        for w in range(inputVars.shape[1]):
            #network.setLowerBound(inputVars[h][w], 0)
            #network.setLowerBound(inputVars[h][w], 1)
            #network.addInequality([epsilon, inputVars[h][w]], [-1, -1], -image[h][w])
            #network.addInequality([epsilon, inputVars[h][w]], [-1, 1], image[h][w])
            network.setLowerBound(inputVars[h][w][0], lb[inputVars[h][w][0]])
            network.setUpperBound(inputVars[h][w][0], ub[inputVars[h][w][0]])
    
    r = np.argsort(-results)
    targets = []
    for b in r[0]:
        targets.append(b)
        for a in range (10):
            if  (a not in targets):
                network1 = copy.deepcopy(network)
                network1.addInequality([outputVars[a], outputVars[b]], [-1, 1], 0)
                network.saveQuery('queryFile')
                vals, stats = network1.solve(options = options)
                if not vals:
                    #print ("unsat")
                    continue
                else:
                    #output = [vals[a] for a in range (784, 794)]
                    #po= np.argmax(output)
                    return False #po==targetLabel
            
    
    
    return True #po==targetLabel


def topk(network, image, results, epsilon, k = 1):
    
    #epsilon = network.getNewVariable()
    #epsilon = 0.0004
    #network.setLowerBound(epsilon, 0) #ub = 0.9 lb = 0 (0.5)
    #network.setUpperBound(epsilon, 1)
    inputVars = network.inputVars[0][0]#[0]
    outputVars = network.outputVars[0]
    lb = np.clip(image-epsilon, 0, 1)
    ub = np.clip(image+epsilon, 0, 1)
    for h in range(inputVars.shape[0]):
        for w in range(inputVars.shape[1]):
            network.setLowerBound(inputVars[h][w], lb[inputVars[h][w]])
            network.setUpperBound(inputVars[h][w], ub[inputVars[h][w]])
    
    r = np.argsort(-results)[0][:k]
    print (r)
    for a in range (10):
        if  (a not in r):
            for b in r :
                network1 = copy.deepcopy(network)
                network1.addInequality([outputVars[a], outputVars[b]], [-1, 1], 0)
                vals, stats = network1.solve(options = options)
                if not vals:
                    continue
                else:
                    test = np.array( tuple(vals.values()))[:784].reshape(28, 28)*255
                    new_image = Image.fromarray(test)
                    if new_image.mode != 'RGB':
                        new_image = new_image.convert('RGB')
                    new_image.save('new.png')
                    f = open("list_of_examples.txt", "a")
                    f.write(str(a)+" "+ str(b)+"\r")
                    f.close()
                    return False 

    
    
    return True 



if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) != 3:
        raise Exception("usage: python bounds.py <epsilon> <filename>")

    epsilon = float(sys.argv[1])
    filename = sys.argv[2]
    # Load the mnist test data
    _, (x_test, y_test) = keras.datasets.mnist.load_data()

    # Test 1
    print("Fully Connected Network Example")
    #filename = "model.onnx"
    #filename = "modelp1.onnx"
    inputName = 'Placeholder:0'
    outputName = 'y_out:0'
    
   
    numberOfTests = 0
    verifiedFair = 0
    for index in range(100):
        #if index<=2:
         #   continue
        network = Marabou.read_onnx(filename=filename, inputNames=["flatten_input"], outputName = "dense_1")
        #network = Marabou.read_onnx(filename=filename, inputNames=["conv2d_input"], outputName = "dense_1")
        #network = Marabou.read_onnx(filename=filename, inputNames=["flatten_1_input"], outputName = "dense_3")

        #new_image = Image.fromarray(x_test[index])
        #if new_image.mode != 'RGB':
            #new_image = new_image.convert('RGB')
        #new_image.save('actual.png')
        image = x_test[index].flatten() / 255
        targetLabel = y_test[index]
        results = network.evaluateWithoutMarabou([image])
        prediction = np.argmax(results)

        #epsilon = 0.1
        if(prediction != targetLabel):
            print ("Classified Wrong")
            continue
        numberOfTests += 1
        verified =  topk(network, image, results, epsilon)#topk(, k = 3)
        
        if verified:
            verifiedFair += 1
        
        print(f"img {index}, correct label: {targetLabel}, verified: {verified}")
        #if not verified:
         #   break
        
    print("{:.2%} certified fair".format(verifiedFair/ numberOfTests))
   
    end = time.time()
    print (end-start)
    print (numberOfTests)
    exit(0)



