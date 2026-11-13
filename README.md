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
