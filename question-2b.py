#Using SVM
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
gls_df=pd.read_csv("./glass.csv")
X=gls_df.drop(['Type'],axis=1)
y=gls_df['Type']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
classifier=GaussianNB()
classifier.fit(X,y)
y_pred=classifier.predict(X_test)
from sklearn.svm import SVC
svc = SVC(max_iter=1000)
X_trainsvc, X_testsvc, y_trainsvc, y_testsvc = train_test_split(X, y, test_size = 0.2, random_state = 0)
svc.fit(X_trainsvc, y_trainsvc)
Y_predsvc = svc.predict(X_testsvc)
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_testsvc,Y_predsvc))
from sklearn.metrics import classification_report
print(classification_report(y_testsvc, y_pred))
matrix = gls_df.corr()
print(matrix)
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()
