import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
t_train=pd.read_csv("./train.csv")
t_test=pd.read_csv("./test.csv")

#Replacing Sex and Embarked with numerical values
t_train['Sex'] = t_train['Sex'].replace(["female", "male"], [0, 1])
t_train['Embarked'] = t_train['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
matrix = t_train.corr()
print(matrix)

#Heatmap showing the correlation between variables
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()

# We need to keep the "Sex" feature because it has high correlation with survived which is the feature to be found
sns.histplot(data=t_train, x="Survived", hue="Sex")
plt.show()
classifier=GaussianNB()
t_train.dropna(axis=0,inplace=True)
t_test['Sex'] = t_train['Sex'].replace(["female", "male"], [0, 1])
t_test['Embarked'] = t_train['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
t_test.dropna(axis=0,inplace=True)
x=t_train.loc[:,['Age', 'Embarked', 'Fare', 'Parch', 'Sex', 'SibSp']]
y=t_train['Survived']
x_test=t_test.loc[:,['Age', 'Embarked', 'Fare', 'Parch', 'Sex', 'SibSp']]
y_test=t_test
from sklearn.metrics import accuracy_score
classifier.fit(x,y)
y_pred=classifier.predict(x_test)
print('accuracy is',accuracy_score(y[:13], y_pred))
