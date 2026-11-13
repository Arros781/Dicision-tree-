Theory: What Are Decision Trees?
Decision Trees are supervised machine learning models used for classification and regression tasks. They work by recursively splitting the data into subsets based on feature values, creating a tree‑like structure.

Key Concepts
Root node: the starting point of the tree.

Internal nodes: represent decisions based on features (e.g., Age < 10).

Branches: represent the outcome of a decision (True/False).

Leaf nodes (terminal nodes): final predictions (class label or value).

Splitting criterion: a rule to decide the best feature and threshold for splitting (e.g., Gini impurity, entropy, MSE).

Depth: the length of the longest path from root to leaf.

Overfitting: trees can become too complex and memorize the training data. Regularization techniques (max depth, min samples per split) help prevent this.

How It Works
Start at the root node with all data.
Find the best split using a criterion (e.g., minimize Gini impurity).
Split the data into two groups.
Repeat steps 2–3 for each new node.
Stop when a stopping condition is met (max depth reached, pure node, etc.).
Assign a prediction to each leaf node.
Advantages
Easy to understand and visualize.
Require little data preprocessing.
Handle both numerical and categorical features.
Non‑parametric — no assumptions about data distribution.
Disadvantages
Prone to overfitting.
Can be unstable (small data changes → different tree).
May not perform as well as more complex models.
Implementation Notes: Decision Tree — Titanic Survival Prediction
Goal. Build an interpretable binary classifier to predict passenger survival using a local Kaggle dataset. No external URLs: the notebook must work offline in the Kaggle environment.

Dataset
Local CSV in the Kaggle environment (/kaggle/input/datasets/heptapod/titanic/train_and_test2.csv).

The file is already numeric: categories are encoded (Sex → 0/1, Embarked → numbers), there are no raw strings. Extra columns (zero.*) were present due to repeated get_dummies and had to be filtered out. The target column is named 2urvived (not Survived), and the ID is Passengerid (not PassengerId).

<img width="1697" height="607" alt="Снимок экрана 2026-11-13 182217" src="https://github.com/user-attachments/assets/ac856e47-001d-4a8e-b4fd-16ff3b6b9fee" />

Step 1. Cleaning
Drop non-informative columns (only existing ones, with errors='ignore').
Remove all columns starting with zero (artifacts of repeated one-hot encoding).
Separate X and y.

Step 2. Split and model
train_test_split(..., test_size=0.2, random_state=42, stratify=y).
DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5).
model.fit(X_train, y_train).
<img width="633" height="257" alt="Снимок экрана 2026-11-13 182252" src="https://github.com/user-attachments/assets/0f4e5a27-56a6-405c-9603-e1978d3a5380" />

Step 3. Evaluation
predict, accuracy_score, classification_report, confusion_matrix.
Heatmap via seaborn.
<img width="698" height="595" alt="Снимок экрана 2026-11-13 182516" src="https://github.com/user-attachments/assets/184b3f86-cfbc-4f12-90f4-ca9039f487cb" />


Step 4. Interpretation
plot_tree: structure of the tree (root: Sex <= 0.5, leaf values with samples and class).
feature_importances_: top features (Sex, Pclass, Fare, Age, Sibsp, Parch, Embarked).
<img width="1726" height="888" alt="Снимок экрана 2026-11-13 182353" src="https://github.com/user-attachments/assets/61085eaf-ac98-4874-921f-bd855df6f597" />
<img width="335" height="240" alt="Снимок экрана 2026-11-13 182510" src="https://github.com/user-attachments/assets/d60c04d8-0c94-4e2b-baa6-c38cb2297fab" />


Summary
A decision tree was trained on a pre-encoded local Titanic dataset. Regularization parameters (max_depth, min_samples_split, min_samples_leaf) were used to limit complexity. Quality was checked on a held-out test set with accuracy, a classification report and a confusion matrix; the resulting tree and feature importance table provide an interpretable view of the model.

