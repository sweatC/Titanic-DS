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
# plt.show()

# correlating numerical and ordinal features
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# plt.show()

# correlating categorical features
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# plt.show()

# correlating categorical and numerical features
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# plt.show()

# correcting by dropping features
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
print(SEPARATOR)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print("Title feature:\n{}\n{}".format(pd.crosstab(train_df['Title'], train_df['Sex']), SEPARATOR))

for dataset in combine:
    dataset["Title"] = dataset["Title"].replace(['Don', 'Lady', 'Countess', 'Capt', 'Col', 'Don',
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], "Rare")

print("Correlation between Title and Survived:\n{}\n{}".format(
    train_df[["Title", "Survived"]].groupby(["Title"], as_index=False).mean(), SEPARATOR))

# converting categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)

print("After convertation:\n{}\n{}".format(train_df.head(), SEPARATOR))

# dropping out Name and PassengerId features
train_df = train_df.drop(["Name", "PassengerId"], axis=1)
test_df = test_df.drop(["Name"], axis=1)
combine = [train_df, test_df]
print("Shapes after dropping:\ntraining set {}\ntest set {}\n{}".format(train_df.shape, test_df.shape, SEPARATOR))
