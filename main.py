# data analysis
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model  import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

SEPARATOR = '_'*40

train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

combine = [train_df, test_df]

print("Features available in the dataset:\n{}\n{}\n".format(train_df.columns.values, SEPARATOR))

print("Preview the data:\n{}\n{}\n".format(train_df.head(), SEPARATOR))

# data types for various features
print("\n{}\n{}".format(train_df.info(null_counts=False), SEPARATOR))
print("\n{}\n{}".format(test_df.info(null_counts=False), SEPARATOR))

# distribution of numerical features
print("\n{}\n{}".format(train_df.describe(), SEPARATOR))

# distribution of categorical features
print("\n{}\n{}".format(train_df.describe(include=['O']), SEPARATOR))

# analyze Pclass feature
print("\n{}\n{}".format(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
                        .sort_values(by='Survived', ascending=False), SEPARATOR))
# analyze Sex feature
print("\n{}\n{}".format(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
                        .sort_values(by='Survived', ascending=False), SEPARATOR))
# analyze Sib/Sp feature
print("\n{}\n{}".format(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
                        .sort_values(by='Survived', ascending=False), SEPARATOR))
# analyze Parch feature
print("\n{}\n{}".format(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
                        .sort_values(by='Survived', ascending=False), SEPARATOR))

# correlating numerical features
grid = sns.FacetGrid(train_df, col='Survived')
grid.map(plt.hist, 'Age', bins=20)
plt.show()

# correlating numerical and ordinal features
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

# correlating categorical features
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

# correlating categorical and numerical features
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()
