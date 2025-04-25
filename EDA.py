import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv('titanic_cleaned.csv')

# Generate summary statistics (mean, median, std, etc.)
print("Summary Statistics:")
print(df.describe(include='all'))
print("\nMedian Age:", df['Age'].median())
print("Median Fare:", df['Fare'].median())

# Create histograms and boxplots for numeric features
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Use pairplot/correlation matrix for feature relationships
# Select only numeric columns for pairplot and correlation
pairplot_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived']
sns.pairplot(df[pairplot_cols])
plt.show()

corr = df[pairplot_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Identify patterns, trends, or anomalies in the data
# Example: Survival rate by Sex
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex (0=male, 1=female)')
plt.show()

# Example: Survival rate by Pclass
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

