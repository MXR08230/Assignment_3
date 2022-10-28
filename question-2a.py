import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
gls_df=pd.read_csv('.\glass.csv')
X=gls_df.drop(['Type'],axis=1)
y=gls_df['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
classifier=GaussianNB()
classifier.fit(X,y)
y_pred=classifier.predict(X_test)
print('accuracy is',accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))