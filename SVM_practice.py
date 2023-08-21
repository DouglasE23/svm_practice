# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC

# Fetch Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Read dataset to pandas dataframe
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
iris_data = pd.read_csv(url, names=colnames)

# Extract the features (X) and target variable (y)
X = iris_data.drop('Class', axis=1)
y = iris_data['Class']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) #70% training and 30% test

# Create and train model a
clf1 = svm.SVC(kernel='poly', degree=3)
clf1.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred_a = clf1.predict(X_test)

# View confusion matrix
cm = confusion_matrix(y_test, y_pred_a)
print(cm)

# Create and train model b
clf2 = svm.SVC(kernel='rbf')
clf2.fit(X_train, y_train)

# Predict the labels for the test data
y_pred_b = clf2.predict(X_test)

# View confusion matrix
cm2 = confusion_matrix(y_test, y_pred_b)
print(cm)

# Create and train model c
clf3 = svm.SVC(kernel='sigmoid')
clf3.fit(X_train, y_train)

# Predict the labels for the test data
y_pred_c = clf3.predict(X_test)

# View confusion matrix
cm = confusion_matrix(y_test, y_pred_c)
print(cm)

# View quality metrics
accuracy_rbf = accuracy_score(y_test, y_pred_a)
accuracy_poly = accuracy_score(y_test, y_pred_b)
accuracy_sigmoid = accuracy_score(y_test, y_pred_c)

# Calculate precision
precision_rbf = precision_score(y_test, y_pred_a, average='weighted')
precision_poly = precision_score(y_test, y_pred_b, average='weighted')
precision_sigmoid = precision_score(y_test, y_pred_c, average='weighted')

# Calculate recall
recall_rbf = recall_score(y_test, y_pred_a, average='weighted')
recall_poly = recall_score(y_test, y_pred_b, average='weighted')
recall_sigmoid = recall_score(y_test, y_pred_c, average='weighted')

# Calculate F1 score
f1_rbf = f1_score(y_test, y_pred_a, average='weighted')
f1_poly = f1_score(y_test, y_pred_b, average='weighted')
f1_sigmoid = f1_score(y_test, y_pred_c, average='weighted')

# Print the metrics
print("Accuracy - RBF Kernel:", accuracy_rbf)
print("Accuracy - Polynomial Kernel:", accuracy_poly)
print("Accuracy - Sigmoid Kernel:", accuracy_sigmoid, '\n')
print("Precision - RBF Kernel:", precision_rbf)
print("Precision - Polynomial Kernel:", precision_poly)
print("Precision - Sigmoid Kernel:", precision_sigmoid, '\n')
print("Recall - RBF Kernel:", recall_rbf)
print("Recall - Polynomial Kernel:", recall_poly)
print("Recall - Sigmoid Kernel:", recall_sigmoid, '\n')
print("F1 Score - RBF Kernel:", f1_rbf)
print("F1 Score - Polynomial Kernel:", f1_poly)
print("F1 Score - Sigmoid Kernel:", f1_sigmoid)