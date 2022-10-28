#Using NaiveByes
import pandas as pd
from sklearn.naive_bayes import GaussianNB
gls_df=pd.read_csv("./glass.csv")
X=gls_df.drop(['Type'],axis=1)
y=gls_df['Type']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
classifier=GaussianNB()
classifier.fit(X,y)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_test,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))