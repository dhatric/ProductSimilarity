import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


from matplotlib.colors import ListedColormap

# ==========================Start Prepare Data========================================================================
# Zoo Dataset is originally provided by the UCI Machine Learning Repository.
animal = pd.read_csv('data/zoo.csv')
print(animal.head())
# check if there is null value
print(animal.isnull().sum())
# Print unique class types
print(animal.class_type.unique())
# Check for duplicate animals
duplicates = animal.animal_name.value_counts()
print(duplicates[duplicates > 1])
# select these duplicates frog to see the data
print(animal.loc[animal['animal_name'] == 'frog'])
# find that one frog is venomous and another one is not
# change the venomous one into venoumous_frog to seperate 2 kinds of frog
animal['animal_name'][(animal.venomous == 1) & (animal.animal_name == 'frog')] = "venoumous_frog"

ani_class = pd.read_csv('data/class.csv')
df = pd.merge(animal, ani_class, how='left', left_on='class_type', right_on='Class_Number')
print(df.head())

# use seaborn to plot the number of each class_type
sns.factorplot('Class_Type', data=df, kind="count", size=5, aspect=2)

# ==========================End Prepare Data===========================================================================

X = animal.iloc[:,1:17]
y = animal.iloc[:,17]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# =======================Decision Tree Classifier===================================================================
# Declare and train the model
clf = DecisionTreeClassifier(random_state = 0,criterion='gini')
clf.fit(X_train, y_train)

y_pred_DecisionTreeClassifier = clf.predict(X_test)
scores = []
score = accuracy_score(y_pred_DecisionTreeClassifier,y_test)
scores.append(score)


# =============================Linear Kernal =========================================================================
# Declare the model
svm = SVC(kernel='linear', C=0.2, random_state=0)

# Train the model
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

#Get Accuracy Score
score = accuracy_score(y_pred_svm,y_test)
scores.append(score)




