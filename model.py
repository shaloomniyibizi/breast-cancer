### Import Libraries


import pandas as pd
import warnings
import pickle
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings('ignore')

### Import Datset
df = pd.read_csv("wiscons.csv")
# we change the class values (at the column number 2) from B to 0 and from M to 1
df.iloc[:,1].replace('B', 0,inplace=True)
df.iloc[:,1].replace('M', 1,inplace=True)

### Splitting Data

X = df[['texture_mean','area_mean','concavity_mean','area_se','concavity_se','fractal_dimension_se','smoothness_worst','concavity_worst', 'symmetry_worst','fractal_dimension_worst']]
y = df['diagnosis']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

#### Data Preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)


# Train two different classification models
# Train first using logistic regression classification models
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)
pred1 = model1.predict(x_test)
accuracy1 = accuracy_score(y_test, pred1)

# Train the second model using a RandomForestClassifier algorithm
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
pred2 = model2.predict(x_test)
accuracy2 = accuracy_score(y_test, pred2)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("Confusion Matrix : \n\n" , confusion_matrix(pred1,y_test))
print("Classification Report : \n\n" , classification_report(pred1,y_test),"\n")

pickle.dump(scaler, open('scaler.pkl', 'wb'))
scaler = pickle.load(open('scaler.pkl','rb'))
print("scaller: ", scaler)

pickle.dump(model1, open('model1.pkl', 'wb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
print(model1)
print("Model 1 Accuracy:", accuracy1)

pickle.dump(model2, open('model2.pkl', 'wb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
print(model2)
print("Model 2 Accuracy:", accuracy2)

pickle.dump(accuracy1, open('accuracy1.pkl','wb'))
pickle.dump(accuracy2, open('accuracy2.pkl','wb'))