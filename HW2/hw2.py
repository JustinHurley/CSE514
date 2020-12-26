#HOW TO RUN CODE
#This code was run using an Anaconda environment on Python 3.8.3
#Running the code returns the results for each of the algorithms in the format:
#NAME
#Evaluation Metrics
#Running time

#Homework 2
#I would like to give credit to https://scikit-learn.org/stable/index.html for helping me understand all their methods
import numpy as np
import csv 
import math
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import time
import matplotlib.pyplot as plt

#given a file path, reads file and returns matrix, row is data point, col is specific val in point
#Used https://realpython.com/python-csv/ to figure out how to read in csv files
#For ref after formatting:
#there are 683 points
#there are either 614 test and 69 train or 615 test and 68 train
def format_data(file_path):
  data = []
  with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      if '?' not in row: #removes the incomplete data points
        data.append(row) #removing id as not an important value
  data = np.array(data).astype(int)[:,1:] #gets rid of the id column as not needed info
  X = data[:,:-1]
  y = data[:,-1]
  #1 is malignant and 0 is benign
  for i in range(len(y)):
    if y[i] == 4:
      y[i] = 1
    elif y[i] == 2:
      y[i] = 0
    else:
      raise Exception("point",i,"in label set is",y[i],"cannot be labelled properly")
  return X, y

#cuts the preformatted data up and returns an array of size 10 
#returns a dict containing 10 diff sets for cross validation
def ten_fold_cross_validation(X, y):
  kf = KFold(n_splits = 10)
  data = []
  for train_index, test_index in kf.split(X):
    data.append({'X_train': X[train_index], 'X_test': X[test_index], 'y_train': y[train_index], 'y_test': y[test_index]})
  return data

#given a dict of the X and y points, unpacks them all and returns them
def unpack_data(data):
  X_train = data['X_train']
  y_train = data['y_train']
  X_test = data['X_test']
  y_test = data['y_test']
  return X_train, y_train, X_test, y_test

def run_algorithm(algorithm, data):
  X_train, y_train, X_test, _ = unpack_data(data)
  algorithm.fit(X_train, y_train)
  return algorithm.predict(X_test)

#aggregates the data and runs the code with the algorithms
def run():
  file_path = 'breast-cancer-wisconsin.data'
  X, y = format_data(file_path)
  dataset = ten_fold_cross_validation(X, y)
  #instantiation of algorithms
  knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
  decision_tree = DecisionTreeClassifier(random_state=0, criterion='entropy')
  random_forest = RandomForestClassifier(max_depth=1,random_state=0,criterion='entropy')
  svm_poly = svm.SVC(kernel='poly',degree=2)
  svm_gauss = svm.SVC(kernel='rbf')
  neural_network = MLPClassifier(solver='sgd',activation='relu',max_iter=1000)
  algorithms = {"Nearest-Neighbor": knn, "Decision Tree": decision_tree, "Random Forest": random_forest, "SVM Polynomial Kernel": svm_poly, "SVM Gaussian Kernel": svm_gauss, "Neural Network": neural_network}
  
  results = []
  #for each of the algorithms
  for algo_name, algo in algorithms.items():
    curr = algo
    #used to track performance over 10 folds
    eval_sum = {'accuracy': 0, 'precision': 0, 'sensitivity': 0, 'specificity': 0}
    tic = time.perf_counter()
    #run through each of the 10 folds
    for i in range(len(dataset)):
      curr_eval = eval(run_algorithm(curr, dataset[i]), dataset[i]['y_test'])
      for key in curr_eval.keys():
        eval_sum[key] += curr_eval[key]
    toc = time.perf_counter()
    algo_eval = {}
    for key, val in eval_sum.items():
      algo_eval[key] = val/10
    results.append([algo_name, algo_eval, toc-tic])
  return results
    

#evaluates the predicted and actual results for a given fold of data
def eval(pred, actual):
  if(len(pred) != len(actual)):
    raise Exception("predicted and actual labels are different lengths")
  total = len(pred)
  TP, FP, TN, FN, P_pred, P_actual, N_pred, N_actual = 0, 0, 0, 0, 0, 0, 0, 0
  for i in range(total):
    p = pred[i]
    a = actual[i]
    #counts predicted P and N
    if(p == 1):
      P_pred += 1
    else:
      N_pred += 1
    #counts actual P and N
    if(a == 1):
      P_actual += 1
    else:
      N_actual += 1
    
    if(a == 1 and p == 1): #true positive
      TP += 1
    elif(a == 1 and p == 0): #false negative
      FN += 1
    elif(a == 0 and p == 1): #false positive
      FP += 1
    else:
      TN += 1
  accuracy = (TP + TN)/total
  precision = TP/P_pred
  sensitivity = TP/P_actual #AKA recall AKA TP rate
  specificity = TN/N_actual #AKA TN rate
  results = {'accuracy': accuracy, 'precision': precision, 'sensitivity': sensitivity, 'specificity': specificity}
  return results

#this line runs the code
results = run()
for a in results:
  for b in a:
    print(b)

