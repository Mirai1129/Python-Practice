import csv
import math
import random

TRAINING_DATA_FILE_NAME: str = "train.csv"
TRAINING_LABELS_FILE_NAME: str = "train_gt.csv"
VALIDATION_DATA_FILE_NAME: str = "val.csv"

class AdvancedKNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self.train_X = []
        self.train_y = []
        self.feature_mins = []
        self.feature_maxs = []

        # Advanced feature weights for importance
        self.feature_weights = []

    def _custom_distance(self, x, y):
        """Advanced distance calculation with feature weighting"""
        weighted_dist = sum(
            weight * ((a - b) ** 2)
            for a, b, weight in zip(x, y, self.feature_weights)
        )
        return math.sqrt(weighted_dist)

    def _polynomial_features(self, features):
        """Create polynomial features for non-linear relationships"""
        poly_features = features.copy()

        # Quadratic terms
        for i in range(len(features)):
            poly_features.append(features[i] ** 2)

        # Interaction terms
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                poly_features.append(features[i] * features[j])

        return poly_features

    def _feature_importance_estimation(self, X, y):
        """Estimate feature importance using simple variance analysis"""
        feature_importance = []
        label_classes = set(y)

        for feature_idx in range(len(X[0])):
            feature_values = [sample[feature_idx] for sample in X]
            class_means = {
                label: sum(
                    val for val, lab in zip(feature_values, y) if lab == label
                ) / y.count(label)
                for label in label_classes
            }

            # Calculate variance between class means
            variance = sum(
                abs(mean1 - mean2)
                for label1, mean1 in class_means.items()
                for label2, mean2 in class_means.items()
                if label1 != label2
            )
            feature_importance.append(variance)

        # Normalize weights
        total_importance = sum(feature_importance)
        self.feature_weights = [
            imp / total_importance if total_importance > 0 else 1
            for imp in feature_importance
        ]

    def normalize_data(self, data):
        """Advanced normalization with robust scaling"""
        if not self.feature_mins:
            self.feature_mins = [min(col) for col in zip(*data)]
            self.feature_maxs = [max(col) for col in zip(*data)]

        normalized_data = []
        for row in data:
            normalized_row = [
                (x - min_val) / (max_val - min_val + 1e-8)
                for x, min_val, max_val in zip(row, self.feature_mins, self.feature_maxs)
            ]
            normalized_data.append(normalized_row)

        return normalized_data

    def preprocess_data(self, data, labels=None):
        """Advanced preprocessing with feature engineering"""
        processed_data = []
        processed_labels = []

        for i, row in enumerate(data):
            features = []

            # Advanced categorical and numerical feature extraction
            features.append(1 if row[0] == 'Male' else 0)
            features.append(1 if row[1] == 'Yes' else 0)
            features.append(1 if row[2] == 'Yes' else 0)
            features.append(1 if row[3] == 'Yes' else 0)

            # Tenure features with advanced encoding
            tenure = float(row[4])
            features.append(tenure / 72.0)  # Normalized tenure
            features.append(1 if tenure <= 12 else 0)  # Short-term customer
            features.append(1 if tenure > 36 else 0)  # Long-term customer

            # Enhanced service features
            features.extend([
                1 if row[7] == 'Fiber optic' else 0,
                1 if row[5] == 'Yes' and row[7] == 'Fiber optic' else 0
            ])

            # Contract and payment complexity
            features.extend([
                1 if row[14] == 'Month-to-month' else 0,
                1 if row[16] == 'Electronic check' else 0
            ])

            # Monetary features with complexity
            monthly_charges = float(row[17])
            total_charges = float(row[18]) if row[18] and row[18] != '' else 0.0

            features.extend([
                monthly_charges / 120.0,  # Normalized monthly charges
                total_charges / 8000.0,   # Normalized total charges
                1 if monthly_charges > 80 else 0,  # High monthly charges
                1 if total_charges > 4000 else 0   # High total charges
            ])

            processed_features = self._polynomial_features(features)
            processed_data.append(processed_features)

            if labels is not None and i < len(labels):
                processed_labels.append(labels[i])

        processed_data = self.normalize_data(processed_data)

        # Estimate feature importance
        if labels:
            self._feature_importance_estimation(processed_data, processed_labels)

        return (processed_data, processed_labels) if labels is not None else processed_data

    def weighted_voting(self, k_nearest):
        """Advanced weighted voting with distance-based importance"""
        votes = {}
        for dist, label in k_nearest:
            weight = 1 / (dist + 1e-8)
            votes[label] = votes.get(label, 0) + weight
        return max(votes.items(), key=lambda x: x[1])[0]

    def predict(self, X):
        """Advanced prediction with custom distance and weighted voting"""
        predictions = []

        for sample in X:
            # Use custom distance calculation with feature weights
            distances = [
                (self._custom_distance(sample, x_train), label)
                for x_train, label in zip(self.train_X, self.train_y)
            ]

            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            majority_class = self.weighted_voting(k_nearest)
            predictions.append(majority_class)

        return predictions

    def cross_validate(self, X, y, k_values=[3, 5, 7, 9], n_folds=5):
        """Simple cross-validation to find best k"""
        best_k = self.k
        best_accuracy = 0

        for k in k_values:
            self.k = k
            fold_size = len(X) // n_folds
            accuracies = []

            for fold in range(n_folds):
                start = fold * fold_size
                end = (fold + 1) * fold_size

                # Split data
                val_X = X[start:end]
                val_y = y[start:end]
                train_X = X[:start] + X[end:]
                train_y = y[:start] + y[end:]

                # Train and predict
                self.fit(train_X, train_y)
                predictions = self.predict(val_X)

                # Calculate accuracy
                accuracy = sum(p == a for p, a in zip(predictions, val_y)) / len(val_y)
                accuracies.append(accuracy)

            avg_accuracy = sum(accuracies) / len(accuracies)

            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_k = k

        self.k = best_k
        return best_k, best_accuracy

    def fit(self, X, y) -> None:
        """
        Training model by using KNN algorithm
        :param X:
        :param y:
        :return:
        """
        if len(X) != len(y):
            raise ValueError(f"Mismatch in data and labels lengths: X has {len(X)} samples, y has {len(y)} labels")

        self.train_X = X
        self.train_y = y


def load_data(filename: str) -> list[list[str]]:
    """
    Load data from CSV file
    :param filename: CSV file name
    :return: List of data rows
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return list(reader)


def save_predictions_to_csv(filename: str, predictions: list) -> None:
    """
    Save predictions to CSV file
    :param filename: Output filename
    :param predictions: List of predictions
    :return: None
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Churn'])
        for pred in predictions:
            writer.writerow([pred])


if __name__ == '__main__':
    # 初始化 KNN 分類器，設定 k 值
    classifier = AdvancedKNNClassifier(k=5)

    # 讀取訓練數據與標籤
    train_data = load_data(TRAINING_DATA_FILE_NAME)
    train_labels = [label[0] for label in load_data(TRAINING_LABELS_FILE_NAME) if len(label) > 0]

    # 預處理訓練數據
    train_X, train_y = classifier.preprocess_data(train_data, train_labels)

    # 讀取驗證數據
    val_data = load_data(VALIDATION_DATA_FILE_NAME)
    val_X = classifier.preprocess_data(val_data)

    # 訓練模型
    classifier.fit(train_X, train_y)

    # 預測驗證數據
    val_predictions = classifier.predict(val_X)

    # 將驗證結果儲存為 val_pred.csv
    save_predictions_to_csv('val_pred.csv', val_predictions)

    # 預測測試數據
    test_data = load_data('test.csv')
    test_X = classifier.preprocess_data(test_data)
    test_predictions = classifier.predict(test_X)

    # 將測試結果儲存為 test_pred.csv
    save_predictions_to_csv('test_pred.csv', test_predictions)

    # 輸出完成訊息
    print("Predictions for validation and test datasets have been saved to 'val_pred.csv' and 'test_pred.csv'.")
