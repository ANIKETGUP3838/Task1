import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Internship\\Task1.csv')

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns=['Cabin'], inplace=True)

data_encoded = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = data_encoded[features]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Did Not Survive', 'Survived'], yticklabels=['Did Not Survive', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

coefficients = model.coef_[0]
feature_importance = pd.Series(coefficients, index=features).sort_values()
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

data_encoded_for_prediction = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
X_all = data_encoded_for_prediction[features]
X_all = scaler.transform(X_all)

survival_predictions = model.predict(X_all)
survival_probabilities = model.predict_proba(X_all)

results = pd.DataFrame({
    'PassengerId': data['PassengerId'],
    'Survived': survival_predictions,
    'Survival Probability': survival_probabilities[:, 1],
    'Non-Survival Probability': survival_probabilities[:, 0]
})

results.to_csv('titanic_all_passengers_predictions_Task1.csv', index=False)







