import numpy as np
from collections import Counter

# need to know which feature to use as a splitting point
# split point (numerically, e.g. how do you know to split at n < 5)
# need to know when to stop splitting

# entropy = messiness of data (high entropy means the data is more spread out or disorganized)

class Node:
    # each node either stores a split rule or a final prediction
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature # feature this node was split on
        self.threshold = threshold # number used to split the feature
        self.left = left # child node for values <= threshold
        self.right = right # child node for values > threshold
        self.value = value # if None then not leaf node

    def is_leaf_node(self):
        # leaf node means this node has a final prediction
        return self.value is not None


class DecisionTreeFromScratch:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        # min_samples_split = minimum samples needed to keep splitting
        self.min_samples_split = min_samples_split
        # max_depth = maximum number of levels the tree can grow
        self.max_depth = max_depth
        # n_features = how many features to check when looking for the best split
        self.n_features = n_features
        self.root=None
    
    # X = input
    # y = target
    def fit(self, X, y):
        # if n_features is not given, use all features
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        # start building the tree from the root node
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # n_samples = number of rows
        # total_features = number of columns
        n_samples, total_features = X.shape
        # n_labels = number of unique classes left in this node
        n_labels = len(np.unique(y))

        # check if the stopping criteria are met
        if (n_samples < self.min_samples_split or depth >= self.max_depth or n_labels == 1):
            # if we stop splitting, predict the most common label
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # total_features = number of features we have
        # self.n_features = number of features we want to consider for splitting
        feature_indices = np.random.choice(total_features, self.n_features, replace=False)

        # find best split
        best_feature, best_threshold = self._get_best_split(X, y, feature_indices)

        # if no split improves the data, turn this node into a leaf
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # split the rows into left and right groups
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        # keep growing the left side of the tree
        left_child = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        # keep growing the right side of the tree
        right_child = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        return Node(best_feature, best_threshold, left_child, right_child)  

    def _most_common_label(self, y):
        # count how many times each label appears
        counter = Counter(y)
        # get the label with the highest count
        most_common = counter.most_common(1)[0]
        return most_common[0]

    def _get_best_split(self, X, y, feature_indices):
        # best_gain tracks which split reduces entropy the most
        best_gain = 0
        split_index, split_threshold = None, None

        # check each feature we randomly selected
        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            # possible split values for this feature
            thresholds = np.unique(X_column)

            # try each possible split value
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                # save this split if it is the best one so far
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold
        
        return split_index, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        # get parent entropy
        parent_entropy = self._entropy(y)

        # create left and right groups using this threshold
        left_indices, right_indices = self._split(X_column, threshold)

        # if one side is empty, this is not a useful split
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
    
        # calculate weighted average entropy of children
        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        entropy_left, entropy_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        # calculate information gain using formula
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _split(self, X_column, split_threshold):
        # left side gets values less than or equal to the threshold
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        # right side gets values greater than the threshold
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices
    
    def _entropy(self, y):
        # calculate entropy
        histogram = np.bincount(y) # bincount returns the number of occurrences of each value in the array 
        probabilities = histogram / len(y)

        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy


    def predict(self, X):
        # predict one row at a time by moving through the tree
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # if this is a leaf node, return the final prediction
        if node.is_leaf_node():
            return node.value

        # choose left or right child based on the split rule
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
