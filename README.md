The code reads in the data from the provided .csv file. 
Preprocessing is needed since some of our data is made up of strings, some columns have missing values, replace some values with floating points, etc.
Uses one hot encoding to convert the "Genre" column into usable variables, and the numeric columns are standardized.

Then splits into a training and test batch, and uses 5-fold cross validation on the training set.
Makes predictions on the test set and calculates the Root Mean Squared Error as a measure of the model’s accuracy on the test set.

The program asks the user for a feature to compare the rest against, then it prints the weights (coefficients) of the Ridge regressor for each other numeric feature.
The weights show their impact (correlation) with the target variable.
