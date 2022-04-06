import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

def median(the_list):
##    sort = sorted(the_list)
##    if len(the_list) % 2 == 1:
##        return sort[len(the_list)// 2]
##    else:
##        return (sort[len(the_list) // 2] + sort[(len(the_list)//2) -1])/2
    return np.median(the_list)
def mean(the_list):
    return sum(the_list)/len(the_list)
    
def variance(the_list, mean):
    return sum((x - mean) ** 2 for x in the_list) / len(the_list)

## some functions and code format was  used from the workshop solutions
world = pd.read_csv('world.csv', encoding = 'ISO-8859-1')
life = pd.read_csv('life.csv', encoding = 'ISO-8859-1')

# Get features
features = []
counter = 0
for x in world:
    if counter > 1:
        features.append(x)

    counter += 1

# Get data
data = world[[x for x in features]]

# Merge to exclude countries that are not in both files and delete irrevelevant columns 
data = pd.merge(data, life, on='Country Code', how='inner')
data.sort_values('Country Code', inplace=True)
class_label = data['Life expectancy at birth (years)']
del data['Country']
del data['Year']
del data['Country Code']
del data['Life expectancy at birth (years)']

# Split data 70% training and 30% testing
x_train, x_test, y_train, y_test = train_test_split(data, class_label, train_size=0.70, test_size=0.30, random_state=200)
  
# Compute medians after disregarding missing values
feature_medians = {}
feature_means = {}
feature_variances = {}

for f in features[1:]:  # To skip country code so start from 1 rather than 0
    fixed_data = [float(x) for x in x_train[f] if x!= '..']
    feature_medians[f] = median(fixed_data)
    feature_means[f] = mean(fixed_data)
    feature_variances[f] = variance(fixed_data, feature_means[f])
    
# Impute data with x_train median for both x_train and x_test
for i in range(len(x_train)):
    for f in features[1:]:
        if x_train.iloc[i][f] == '..':
            x_train.iloc[i][f] = str(feature_medians[f])
            
        if i < len(x_test[f]):
            if x_test.iloc[i][f] == '..':
                x_test.iloc[i][f] = str(feature_medians[f])
            
x_train.astype(float)
x_test.astype(float)

# Normalise the data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# METHOD 1 DECISION TREE
dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print(f"Accuracy of decision tree: {accuracy_score(y_test, y_pred):.3f}")

# METHOD 2 KNN WHERE K=3

# fit for K=3
knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn3.fit(x_train, y_train)
y_pred=knn3.predict(x_test)
print(f"Accuracy of k-nn (k=3): {accuracy_score(y_test, y_pred):.3f}")

# METHOD 3 KNN WHERE K=7

# fit for K=7
knn7 = neighbors.KNeighborsClassifier(n_neighbors=7)
knn7.fit(x_train, y_train)
y_pred=knn7.predict(x_test)
print(f"Accuracy of k-nn (k=7): {accuracy_score(y_test, y_pred):.3f}")

# Create and close csv output file
output = open("task2a.csv",'w')
writer = csv.writer(output)
writer.writerow(['feature', 'median', 'mean', 'variance'])
for f in features[1:]:  # To skip features[0] which is country code
    writer.writerow([f, f'{feature_medians[f]:.3f}', f'{feature_means[f]:.3f}', f'{feature_variances[f]:.3f}'])
output.close()
        




                     
