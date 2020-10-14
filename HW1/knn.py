import numpy as np
import csv
import math

train_file_path = 'HW1/breast-cancer-wisconsin-train.data'
test_file_path = 'HW1/breast-cancer-wisconsin-test.data' 
k = 5

#given a file path, reads file and returns matrix, row is data point, col is specific val in point
#Used https://realpython.com/python-csv/ to figure out how to read in csv files
def file_to_matrix(file_path):
  data = []
  with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      if '?' not in row: #removes the incomplete data points
        data.append(row) #removing id as not an important value
  return np.array(data).astype(int)

#euclidean distance of two n-dim points, a and b
#note: items have been appended to only contain the feature values (no id or classification)
def distance(a, b):
  sum = 0
  if len(a) != len(b):
    raise Exception('ERROR: dimensions of points are not equal!')
  #calculates the distance for each dim
  for i in range(len(a)):
    a_int = a[i]
    b_int = b[i]
    sum += ((a_int - b_int) ** 2)
  return math.sqrt(sum)

#actual implementation of KNN 
#returns data set of id and classification for each
def k_nearest_nodes(k, train, test):
  classified_points = []
  #for each of the points we want to predict
  for point_test in test:
    #for each value in the training set
    distances = [] #will store the distances for each point and val
    for point_train in train:
      #get distance from point_test to each point_train
      #store vals in distances, [distance to point, y of point]
      distance_val = distance(point_test[1:-1], point_train[1:-1])
      dist_and_y = [distance_val, point_train[len(point_train)-1]]
      distances.append(dist_and_y)
    #need to sort the distances
    sorted_distances = sorted(distances, key = lambda x: x[0])
    avg = 0
    for i in range(k):
      avg += sorted_distances[i][1]
    avg /= k
    if avg >= 3:
      classified_points.append([point_test, 4])
    else:
      classified_points.append([point_test, 2])
  return classified_points
      
#run k nearest nodes algo
#note if you want to manually input data, just change the train and test data to the right things
def run_k_nearest_nodes(k, train_file_path, test_file_path):
  train = file_to_matrix(train_file_path)
  test = file_to_matrix(test_file_path)
  classified = k_nearest_nodes(k, train, test)
  print(classified)
  to_txt = [['ID','Predicted','Actual','Correct']]
  correct = 0
  for item in classified:
    predicted = item[1]
    actual = item[0][10]
    if predicted == actual:
      to_txt.append([item[0][0], predicted, actual, 'YES'])
      correct += 1
    else:
      to_txt.append([item[0][0], predicted, actual, 'NO'])
  to_txt.append(['Accuracy:',(correct/len(classified))])
  np.savetxt('kNNC_results.txt', to_txt, delimiter=',', fmt='%s')
  return

run_k_nearest_nodes(k, train_file_path, test_file_path)
