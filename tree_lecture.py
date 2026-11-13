import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load
data = pd.read_csv('/kaggle/input/datasets/heptapod/titanic/train_and_test2.csv')
print(data.columns.tolist())
print(data.head())

# 2. Preprocess: drop ID (правильное имя — Passengerid), остальное проверяем
data = data.drop(['Passengerid'], axis=1, errors='ignore')

# Убираем все zero.* — это артефакты многократного кодирования
data = data[[c for c in data.columns if not c.startswith('zero')]]

X = data.drop('2urvived', axis=1)   # правильное имя целевой переменной
y = data['2urvived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Model
model = DecisionTreeClassifier(random_state=42, max_depth=5,
                               min_samples_split=10, min_samples_leaf=5)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 5. Visualize
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'],
          filled=True, rounded=True, fontsize=9)
plt.title('Decision Tree')
plt.show()

imp = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})\
       .sort_values('importance', ascending=False)
print("Feature importance:")
print(imp)
