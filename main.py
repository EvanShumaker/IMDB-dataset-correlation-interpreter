import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load the dataset from a CSV file
data = pd.read_csv('imdb_top_2000_movies.csv')

# Convert columns that contain numbers but are stored as strings into numeric format
data['Release Year'] = pd.to_numeric(data['Release Year'], errors='coerce')
data['Votes'] = data['Votes'].replace(',', '', regex=True).astype(int)


# Function to convert 'Gross' earnings from string to float, handling millions (M) and thousands (K)
def convert_gross(value):
    if pd.isna(value):
        return np.nan
    value = value.replace('$', '').replace(',', '')
    if 'M' in value:
        return float(value.replace('M', '')) * 1e6
    elif 'K' in value:
        return float(value.replace('K', '')) * 1e3
    return float(value)


data['Gross'] = data['Gross'].apply(convert_gross)

# Fill missing values in 'Metascore' and 'Gross' with the median of their respective columns
data['Metascore'] = pd.to_numeric(data['Metascore'], errors='coerce')
data.fillna({'Metascore': data['Metascore'].median(), 'Gross': data['Gross'].median()}, inplace=True)

# Remove columns that are not useful for regression analysis
data.drop(['Movie Name', 'Director', 'Cast'], axis=1, inplace=True)

# missing value checks
missing_values = data.isna() | (data == 'NA')

# Iterate through each row in the DataFrame to find missing values
for index, row in data.iterrows():
    # checking for missing values
    for column in data.columns:
        if missing_values.loc[index, column]:
            median_value = data[column].median()
            data.at[index, column] = median_value


# Setup preprocessing steps for numeric data: imputation for missing values and scaling
numeric_features = ['Release Year', 'Duration', 'Votes', 'Metascore', 'Gross', 'IMDB Rating']
numeric_transformer = Pipeline(steps=[
    # Fills missing values with the median
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])  # Standardizes features by removing the mean and scaling to unit variance

# print(numeric_features)

# Setup preprocessing steps for categorical data: converting 'Genre' using one-hot encoding
categorical_features = ['Genre']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # Converts categorical variable into dummy/indicator variables

# Function to remove a feature from either numeric_features
def remove_feature(feature_name):
    if feature_name in numeric_features:
        numeric_features.remove(feature_name)
    else:
        print(f"'{feature_name}' not found in numeric_features or categorical_features.")
        return False
    return True


# Prompt the user to input a feature name to remove
valid_feature_removed = False
feature_to_remove = ""
while not valid_feature_removed:
    print("Select one feature to compare the rest to: ", numeric_features)
    feature_to_remove = input()
    valid_feature_removed = remove_feature(feature_to_remove)

# Combine preprocessing steps into a single transformer that handles both numeric and categorical processing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create a regression model pipeline that includes preprocessing and the estimator (Ridge regression)
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', Ridge())])

# Split the data into training and testing sets for model validation
X = data.drop(feature_to_remove, axis=1)
y = data[feature_to_remove]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model using the training data
model.fit(X_train, y_train)

# Evaluate the model using cross-validation to get an estimate of its performance on unseen data
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores)}')

# Use the trained model to make predictions on the test set and calculate the RMSE to assess accuracy
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE: {rmse}')

# Accessing coefficients and intercept from Ridge regressor in the Pipeline
ridge_regressor = model.named_steps['regressor']  # Get the Ridge regressor from the Pipeline
coefficients = ridge_regressor.coef_  # Get the coefficients (weights) of the Ridge regressor
intercept = ridge_regressor.intercept_  # Get the intercept (bias) of the Ridge regressor

# Display the coefficients (weights) and intercept
print("\nCoefficient Weight:")

for feature_name, coef in zip(numeric_features, coefficients):
    print(f"{feature_name}: {coef}")

print(f"Intercept: {intercept}")
