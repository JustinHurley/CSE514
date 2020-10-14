TO RUN THE CODE
Need:
  -	Python 3.8.3 or greater 
  -	Numpy 1.5.0
  -	If it’s required to be installed, the csv and math Python packages
note: you can probably get away with a lower version of Python as long as it's Python 3, and same goes for Numpy as long as it is a relatively recent version

Instructions: 
Note that all the input data files must be in comma-delimited CSV format, and it is assumed that the test data will have the classification value, if it does not, please refer to step 4 as to how to correct the code to accept the test set with no classification.

The data should all be ready to run, the only issue could be a file-pathing issue in which the paths on line 5 and 6 may have to be altered.

The test data comes from the last 30 items in the breast-cancer-wisconsin data set. For some reason some items have the same ID but have different feature values.

  1.	On line 5 set ‘train_file_path’ to desired file path (should be set already)
  2.	On line 6 it says ‘test_file_path,’ put the file path to the test data there (should be set   already)
  3.	On line 7, set the desired value of k, it is currently set to 5
  4.	This step should not be necessary given that I have supplied my own test data. If the test data does not contain the classification value (the last value in the index), then proceed to line 44, and change ‘point_test[1:-1]’ to ‘point_test[1:].’ 
    a.	If the test data also does not contain an ID number, change ‘point_test[1:-1]’ to ‘point_test.’ 
    b.	If there was an issue with having the correct dimensions, it should throw an error saying incorrect dimensions and adjust accordingly, the distance(a,b) function was developed to only take the portion of the data that contains the features, so there should be no ID number and no classification value in the input lists.
  5.	Run the code
  6.	Output should print out the test point array, along with the classified value next to it
  7.	Output should also create a ‘kNCC_results.txt’ file, which shows the ID number, the predicted value, the actual value, and if the algorithm classified correctly. On the bottom of the file is an accuracy measure.	
