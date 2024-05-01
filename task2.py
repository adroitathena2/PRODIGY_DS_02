'''
Perform data cleaning and exploratory data analysis (EDA) on a
dataset of your choice, such as Titanic dataset from Kaggle. 
Explore the relationships between variables and identify patterns
and trends in the data. 
'''
#https://github.com/Apoorva5311/PRODIGY_DS_02/blob/main/Task%202.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv("C:\\Users\\archi\\Documents\\INTERNSHIPS\\PRODIGY INFOTECH\\Task 2\\train.csv")
print(titanic_df.head())

print(titanic_df.isnull().sum())
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

titanic_df.drop('Cabin', axis=1, inplace=True)

titanic_df.dropna(subset=['Embarked'], inplace=True)

print(titanic_df.isnull().sum())

# Summary statistics for numerical columns
print(titanic_df.describe())

# Visualize the distribution of numerical variables
sns.histplot(titanic_df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


sns.countplot(x='Survived', data=titanic_df)
plt.title('Count of Survived (0 = No, 1 = Yes)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()


sns.countplot(x='Survived', hue='Sex', data=titanic_df)
plt.title('Survival Count by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Sex', loc='upper right')
plt.show()


sns.countplot(x='Survived', hue='Pclass', data=titanic_df)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Pclass', loc='upper right')
plt.show()