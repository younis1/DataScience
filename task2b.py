import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif as mutual
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest as KBest
from math import sqrt
import csv

# From task 2a ... preprocess data and split

def median(the_list):
    sort = sorted(the_list)
    if len(the_list) % 2 == 1:
        return sort[len(the_list)// 2]
    else:
        return (sort[len(the_list) // 2] + sort[(len(the_list)//2) -1])/2

def get_feature_medians(features, X_data):
    feature_medians = {}
    for f in features:
        fixed_data = [float(x) for x in X_data[f] if x!= '..']
        feature_medians[f] = median(fixed_data)
    return feature_medians

def impute_with_median(X_data, feature_medians):
    for i in range(len(X_data)):
        for f in feature_medians:
            if X_data.iloc[i][f] == '..':
                X_data.iloc[i][f] = str(feature_medians[f])

    return X_data.astype(float)

def scale_normal_data(X_data):
    return (X_data - X_data.mean())/X_data.std()

def clustering(k, X_data):
    the_cluster = KMeans(n_clusters=k, random_state=200).fit(X_data)
    return the_cluster.labels_ , the_cluster.cluster_centers_
    

def eucl_dist(array1, array2):
    dist = 0
    for i in range(len(array1)):
        dist += (array1[i] - array2[i]) ** 2
    return sqrt(dist)

def mutual_scores(X, y):
    return mutual(X, y, random_state=200, n_neighbors=3)

def high_var(features, X):
    var = list(X.var())
    final = []
    z = var.copy()
    for i in range(4):
        the_max = max(z)
        index = var.index(the_max)
        z.remove(the_max)
        final.append(index)
    return final

world = pd.read_csv('world.csv', encoding = 'ISO-8859-1')
life = pd.read_csv('life.csv', encoding = 'ISO-8859-1')

### Method 1 (Feature engineering)

def mutual_scaled(k):
    global world, life
    
    # Get features
    features = []
    counter = 0

    for x in world:
        if counter > 1:
            features.append(x)
        counter += 1

    # Get data and delete irrelevant columns
    data = world[[x for x in features]]
    data = pd.merge(data, life, on='Country Code', how='inner')
    data.sort_values('Country Code', inplace=True)
    class_label = data['Life expectancy at birth (years)']
    del data['Country']
    del data['Year']
    del data['Country Code']
    del data['Life expectancy at birth (years)']

    X_train, X_test, y_train, y_test = train_test_split(data, class_label, train_size=0.70, test_size=0.30, random_state=200)
    medians = get_feature_medians(features[1:], X_train)

    X_train = impute_with_median(X_train, medians)
    X_train = scale_normal_data(X_train)
    
    X_test  = impute_with_median(X_test, medians)
    X_test = scale_normal_data(X_test)
    
    
    for i in range(1,21):
        for j in range(i+1, 21):
            
            X_train[features[i]+'*'+features[j]] = X_train[features[i]] * X_train[features[j]]
            X_test[features[i]+'*'+features[j]] = X_test[features[i]] * X_test[features[j]]
            
            features.append(features[i]+'*'+features[j])
    
    features.append('cluster label')
    
    X_train_labels, X_train_centers = clustering(k, X_train)
    X_train['cluster label'] = X_train_labels

    cluster_column = []
 
    for index, row in X_test[list(X_test.columns)[0:20]].iterrows():
        dist = []   
        for center in X_train_centers:
            dist.append(eucl_dist(row, center))
        min_index = dist.index(min(dist))
        cluster_column.append(X_train_labels[min_index])
    X_test['cluster label'] = cluster_column
    
    
    model = KBest(mutual, k=4).fit(X_train, y_train)
    
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn3.fit(model.transform(X_train), y_train)
    y_pred = knn3.predict(model.transform(X_test))        
    return accuracy_score(y_test, y_pred)

def mutual_unscaled(k):
    global world, life
    
    # Get features
    features = []
    counter = 0

    for x in world:
        if counter > 1:
            features.append(x)
        counter += 1

    # Get data and delete irrelevant columns
    data = world[[x for x in features]]
    data = pd.merge(data, life, on='Country Code', how='inner')
    data.sort_values('Country Code', inplace=True)
    class_label = data['Life expectancy at birth (years)']
    del data['Country']
    del data['Year']
    del data['Country Code']
    del data['Life expectancy at birth (years)']

    X_train, X_test, y_train, y_test = train_test_split(data, class_label, train_size=0.70, test_size=0.30, random_state=200)
    medians = get_feature_medians(features[1:], X_train)

    X_train = impute_with_median(X_train, medians)
    
    X_test  = impute_with_median(X_test, medians)

    
    
    for i in range(1,21):
        for j in range(i+1, 21):
            
            X_train[features[i]+'*'+features[j]] = X_train[features[i]] * X_train[features[j]]
            X_test[features[i]+'*'+features[j]] = X_test[features[i]] * X_test[features[j]]
            
            features.append(features[i]+'*'+features[j])
    
    features.append('cluster label')
    
    X_train_labels, X_train_centers = clustering(k, X_train)
    X_train['cluster label'] = X_train_labels

    cluster_column = []
 
    for index, row in X_test[list(X_test.columns)[0:20]].iterrows():
        dist = []   
        for center in X_train_centers:
            dist.append(eucl_dist(row, center))
        min_index = dist.index(min(dist))
        cluster_column.append(X_train_labels[min_index])
    X_test['cluster label'] = cluster_column
    
    
    model = KBest(mutual, k=4).fit(X_train, y_train)
    
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn3.fit(model.transform(X_train), y_train)
    y_pred = knn3.predict(model.transform(X_test))        
    return accuracy_score(y_test, y_pred)

def f_scaled(k):
    global world, life
    
    # Get features
    features = []
    counter = 0

    for x in world:
        if counter > 1:
            features.append(x)
        counter += 1

    # Get data and delete irrelevant columns
    data = world[[x for x in features]]
    data = pd.merge(data, life, on='Country Code', how='inner')
    data.sort_values('Country Code', inplace=True)
    class_label = data['Life expectancy at birth (years)']
    del data['Country']
    del data['Year']
    del data['Country Code']
    del data['Life expectancy at birth (years)']

    X_train, X_test, y_train, y_test = train_test_split(data, class_label, train_size=0.70, test_size=0.30, random_state=200)
    medians = get_feature_medians(features[1:], X_train)

    X_train = impute_with_median(X_train, medians)
    X_train = scale_normal_data(X_train)
    
    X_test  = impute_with_median(X_test, medians)
    X_test = scale_normal_data(X_test)
    
    
    for i in range(1,21):
        for j in range(i+1, 21):
            
            X_train[features[i]+'*'+features[j]] = X_train[features[i]] * X_train[features[j]]
            X_test[features[i]+'*'+features[j]] = X_test[features[i]] * X_test[features[j]]
            
            features.append(features[i]+'*'+features[j])
    
    features.append('cluster label')
    
    X_train_labels, X_train_centers = clustering(k, X_train)
    X_train['cluster label'] = X_train_labels

    cluster_column = []
 
    for index, row in X_test[list(X_test.columns)[0:20]].iterrows():
        dist = []   
        for center in X_train_centers:
            dist.append(eucl_dist(row, center))
        min_index = dist.index(min(dist))
        cluster_column.append(X_train_labels[min_index])
    X_test['cluster label'] = cluster_column
    
    
    model = KBest(f_classif, k=4).fit(X_train, y_train)
    
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn3.fit(model.transform(X_train), y_train)
    y_pred = knn3.predict(model.transform(X_test))        
    return accuracy_score(y_test, y_pred)


def f_unscaled(k):
    global world, life
    
    # Get features
    features = []
    counter = 0

    for x in world:
        if counter > 1:
            features.append(x)
        counter += 1

    # Get data and delete irrelevant columns
    data = world[[x for x in features]]
    data = pd.merge(data, life, on='Country Code', how='inner')
    data.sort_values('Country Code', inplace=True)
    class_label = data['Life expectancy at birth (years)']
    del data['Country']
    del data['Year']
    del data['Country Code']
    del data['Life expectancy at birth (years)']

    X_train, X_test, y_train, y_test = train_test_split(data, class_label, train_size=0.70, test_size=0.30, random_state=200)
    medians = get_feature_medians(features[1:], X_train)

    X_train = impute_with_median(X_train, medians)
    
    X_test  = impute_with_median(X_test, medians)

    
    
    for i in range(1,21):
        for j in range(i+1, 21):
            
            X_train[features[i]+'*'+features[j]] = X_train[features[i]] * X_train[features[j]]
            X_test[features[i]+'*'+features[j]] = X_test[features[i]] * X_test[features[j]]
            
            features.append(features[i]+'*'+features[j])
    
    features.append('cluster label')
    
    X_train_labels, X_train_centers = clustering(k, X_train)
    X_train['cluster label'] = X_train_labels

    cluster_column = []
 
    for index, row in X_test[list(X_test.columns)[0:20]].iterrows():
        dist = []   
        for center in X_train_centers:
            dist.append(eucl_dist(row, center))
        min_index = dist.index(min(dist))
        cluster_column.append(X_train_labels[min_index])
    X_test['cluster label'] = cluster_column
    
    
    model = KBest(f_classif, k=4).fit(X_train, y_train)
    
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn3.fit(model.transform(X_train), y_train)
    y_pred = knn3.predict(model.transform(X_test))        
##    print(f'Accuracy of feature engineering: {accuracy_score(y_test, y_pred):.3f}')
    return accuracy_score(y_test, y_pred)

### Method 2 (PCA)
def pca_method():
    
    # Get features
    global world, life
    features = []
    counter = 0

    for x in world:
        if counter > 1:
            features.append(x)
        counter += 1

    # Get data and delete irrelevant columns
    data = world[[x for x in features]]
    data = pd.merge(data, life, on='Country Code', how='inner')
    data.sort_values('Country Code', inplace=True)
    class_label = data['Life expectancy at birth (years)']
    del data['Country']
    del data['Year']
    del data['Country Code']
    del data['Life expectancy at birth (years)']

    X_train, X_test, y_train, y_test = train_test_split(data, class_label, train_size=0.70, test_size=0.30, random_state=200)
    medians = get_feature_medians(features[1:], X_train)

    X_train = impute_with_median(X_train, medians)

    X_test  = impute_with_median(X_test, medians)

    # Many of the PCA functions were used from from https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html
    # Reduce PCA to 4 components
    pca = make_pipeline(preprocessing.StandardScaler(), PCA(n_components=4, random_state=200))
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    pca.fit(X_train, y_train)
    knn.fit(pca.transform(X_train), y_train)
    accuracy_pca = knn.score(pca.transform(X_test), y_test)
    print(f"Accuracy of PCA: {accuracy_pca:.3f}")
    
### Method 3 (First 4 column)
def four_features():
    global world, life
    # Get features
    features = []
    counter = 0

    for x in world:
        if counter > 1 and counter < 7:
            features.append(x)
        counter += 1

    # Get data and delete irrelevant columns
    data = world[[x for x in features]]
    data = pd.merge(data, life, on='Country Code', how='inner')
    data.sort_values('Country Code', inplace=True)
    class_label = data['Life expectancy at birth (years)']
    del data['Country']
    del data['Year']
    del data['Country Code']
    del data['Life expectancy at birth (years)']

    # Split data 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(data, class_label, train_size=0.70, test_size=0.30, random_state=200)

    # Impute median and normalise data
    medians = get_feature_medians(features[1:], X_train)
    
    X_train = impute_with_median(X_train, medians)
    X_train = scale_normal_data(X_train)

    X_test  = impute_with_median(X_test, medians)
    X_test  = scale_normal_data(X_test)

    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn3.fit(X_train, y_train)
    y_pred = knn3.predict(X_test)
    

    print(f"Accuracy of first four features: {accuracy_score(y_test, y_pred):.3f}")
    

f_s = {}
f_u = {}
m_s = {}
m_u = {}
for i in range(3,26):
    f_s[i] = f_scaled(i)
    f_u[i] = f_unscaled(i)
    m_s[i] = mutual_scaled(i)
    m_u[i] = mutual_unscaled(i)

plt.scatter(f_s.keys(), f_s.values())
plt.savefig('task2b_Scaled_FValues')
plt.title("No. of clusters Accuracies using F-values (scaled)")
plt.close()

plt.scatter(f_u.keys(), f_u.values())
plt.savefig('task2b_Unscaled_FValues')
plt.title("No. of clusters Accuracies using F-values (unscaled)")
plt.close()

plt.scatter(m_u.keys(), m_u.values())
plt.title("No. of clusters Accuracies using mutual info. (unscaled)")
plt.savefig('task2b_Scaled_Mutual')
plt.close()

plt.scatter(m_s.keys(), m_s.values())
plt.title("No. of clusters Accuracies using mutual info. (scaled)")
plt.savefig('task2b_Unscaled_Mutual')
plt.close()

print(f'Accuracy of feature engineering: {f_u[8]:.3f}')
pca_method()
four_features()
