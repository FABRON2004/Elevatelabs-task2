# Elevatelabs-task2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

summary = df.describe(include='all')
print(summary)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

plt.figure(figsize=(12, 10))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(df[numeric_cols].dropna())
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Pclass', y='Survived')
plt.title('Survival Rate by Passenger Class')
plt.show()

fig = px.scatter(df, x='Age', y='Fare', color='Survived', title='Fare vs Age colored by Survival')
fig.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Sex', y='Age', hue='Survived', split=True)
plt.title('Age distribution by Sex and Survival')
plt.show()
