import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the train and test datasets from files
train = pd.read_csv('/Users/vu/Data_science/train.csv')
test = pd.read_csv('/Users/vu/Data_science/test.csv')
train.info()
test.info()

# Drop obviously unnecessary columns

train = train.drop(['PassengerId','Name','Ticket','Embarked',
                    'Fare','Cabin'], axis=1)
Passenger_id_test = test['PassengerId']
test = test.drop(['PassengerId','Name','Ticket','Embarked',
                  'Fare','Cabin'], axis=1)
data = [train, test]

train.head()
train.tail()
train.describe()

age_train_mean = train['Age'].mean()
age_train_std = train['Age'].std()
age_test_mean = test['Age'].mean()
age_test_std = test['Age'].std()
age_train_null = train['Age'].isnull().sum()
age_test_null = test['Age'].isnull().sum()

for age in range(train['Age'].size):
    if np.isnan(train['Age'][age]):
        train['Age'][age] = np.random.normal(loc=age_train_mean,
                                             scale=age_train_std, size=None)
for age in range(test['Age'].size):
    if np.isnan(test['Age'][age]):
        test['Age'][age] = np.random.normal(loc=age_test_mean,
                                             scale=age_test_std, size=None)

train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
		

from sklearn.linear_model import LogisticRegression

print ('It still run fine')

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.copy()
X_train.shape, Y_train.shape, X_test.shape

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
Y_pred = pd.Series(Y_pred, name = 'Survived')
df = pd.concat([Passenger_id_test, Y_pred], axis=1)

df.to_csv('/Users/vu/Data_science/Diep_gender_submission.csv', index=False)
