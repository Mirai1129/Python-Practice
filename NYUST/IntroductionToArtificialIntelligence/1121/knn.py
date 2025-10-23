import csv
import math
from typing import Any
# using "typing" as limit user using correct parameter type and return value, if can't just delete the parameter limit
# 用 typing 只是為了限制參數及返回值，如果不能用直接刪除即可，不影響程式碼運行

TRAINING_DATA_FILE_NAME: str = "train.csv"
TRAINING_LABELS_FILE_NAME: str = "train_gt.csv"
VALIDATION_DATA_FILE_NAME: str = "val.csv"


class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self._max_tenure = 72.0
        self._max_monthly_charges = 118.75
        self._max_total_charges = 118.75
        self.train_X = [] # training data
        self.train_y = [] # training label
        self.feature_mins = []
        self.feature_maxs = []

    def normalize_data(self, data: list[list[int | float]]) -> list[list[int]]:
        """
        Normalize data by using min-max scaling
        :param data:
        :return:
        """
        if not self.feature_mins:
            self.feature_mins = [min(col) for col in zip(*data)]
            self.feature_maxs = [max(col) for col in zip(*data)]

        normalized_data = []
        for row in data:
            normalized_row = [
                (x - min_val) / (max_val - min_val) if max_val != min_val else 0
                for x, min_val, max_val in zip(row, self.feature_mins, self.feature_maxs)
            ]
            normalized_data.append(normalized_row)

        return normalized_data

    def preprocess_data(self,
                        data: list[list[str]],
                        labels: list[str] = None) -> tuple[list[list[int | Any]], list[Any]] | list[list[int | Any]]:
        """
        Convert categorical and numerical data to appropriate format
        :param data:
        :param labels:
        :return:
        """
        processed_data = []
        processed_labels = []

        if labels is not None:
            labels = labels[:len(data)]

        for i, row in enumerate(data):
            features = []

            # shorter presentation of the features
            # features = [1 if row[0] == 'Male' else 0, 1 if row[1] == 'Yes' else 0, 1 if row[2] == 'Yes' else 0,
            #                         1 if row[3] == 'Yes' else 0, float(row[4]) / self._max_tenure, 1 if row[5] == 'Yes' else 0]

            # Convert categorical variables to numerical
            # Gender
            features.append(1 if row[0] == 'Male' else 0)

            # SeniorCitizen
            features.append(1 if row[1] == 'Yes' else 0)

            # Partner
            features.append(1 if row[2] == 'Yes' else 0)

            # Dependents
            features.append(1 if row[3] == 'Yes' else 0)

            # Tenure - divide max tenure as sample
            features.append(float(row[4]) / self._max_tenure)

            # PhoneService
            features.append(1 if row[5] == 'Yes' else 0)

            # MultipleLines (Yes, No, No phone service)
            if row[6] == 'Yes':
                features.append(1)
            elif row[6] == 'No':
                features.append(0)
            else:
                features.append(-1)

            # InternetService
            if row[7] == 'Fiber optic':
                features.extend([1, 0])
            elif row[7] == 'DSL':
                features.extend([0, 1])
            else:
                features.extend([0, 0])

            # Online services
            online_services = row[8:12]
            for service in online_services:
                if service == 'Yes':
                    features.append(1)
                elif service == 'No':
                    features.append(0)
                else:  # No internet service
                    features.append(-1)

            # Streaming services
            streaming_services = row[12:14]
            for service in streaming_services:
                if service == 'Yes':
                    features.append(1)
                elif service == 'No':
                    features.append(0)
                else:  # No internet service
                    features.append(-1)

            # Contract
            if row[14] == 'Month-to-month':
                features.extend([1, 0, 0])
            elif row[14] == 'One year':
                features.extend([0, 1, 0])
            else:  # Two year
                features.extend([0, 0, 1])

            # PaperlessBilling
            features.append(1 if row[15] == 'Yes' else 0)

            # PaymentMethod
            if row[16] == 'Electronic check':
                features.extend([1, 0, 0, 0])
            elif row[16] == 'Mailed check':
                features.extend([0, 1, 0, 0])
            elif row[16] == 'Bank transfer (automatic)':
                features.extend([0, 0, 1, 0])
            else:  # Credit card (automatic)
                features.extend([0, 0, 0, 1])

            # MonthlyCharges
            features.append(float(row[17]) / self._max_monthly_charges)  # Assuming max is around 118.75

            # TotalCharges
            try:
                features.append(float(row[18]) / self._max_total_charges)
            except ValueError:
                features.append(0.0)

            processed_data.append(features)

            if labels is not None and i < len(labels):
                processed_labels.append(labels[i])

        processed_data = self.normalize_data(processed_data)

        if labels is not None:
            return processed_data, processed_labels
        return processed_data

    @staticmethod
    def _get_euclidean_distance(x: list[int] | tuple[int] | int, y: list[int] | tuple[int] | int) -> float:
        """
        Calculate Euclidean distance between two points
        :param x: x position for the coordinates
        :param y: y position for the coordinates
        :return:
        """
        # check x and y if single integer
        if isinstance(x, int):
            x = [x]
        if isinstance(y, int):
            y = [y]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

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

    def predict(self, X) -> list:
        """
        Make X prediction with detailed logging
        :param X:
        :return:
        """
        predictions = []

        for sample_idx, x in enumerate(X):
            distances = []
            for idx, x_train in enumerate(self.train_X):
                dist = self._get_euclidean_distance(x, x_train)

                # checking list if over of range
                if idx < len(self.train_y):
                    distances.append((dist, self.train_y[idx]))
                else:
                    print(f"Warning: Label not found for index {idx}")

            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]

            # Count votes for each class
            votes = {}
            for _, label in k_nearest:
                votes[label] = votes.get(label, 0) + 1

            # Get the majority class, just like fruits and animals
            majority_class = max(votes.items(), key=lambda x: x[1])[0]
            predictions.append(majority_class)

            print(f"\nPrediction for Sample {sample_idx + 1}:")
            print("K-Nearest Neighbors Details:")
            for i, (dist, label) in enumerate(k_nearest, 1):
                print(f"  Neighbor {i}: Distance = {dist:.4f}, Label = {label}")

            print("\nVote Counts:")
            for label, count in votes.items():
                print(f"  {label}: {count} votes")

            print(f"\nPredicted Class: {majority_class}")
            print("-" * 50)

        return predictions


def load_data(filename: str) -> list[list[str]]:
    """
    Load training data from CSV file
    :param filename: suggest using csv file name
    :return:
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return list(reader)


def save_predictions_to_csv(filename: str, predictions: list) -> None:
    """
    Save predictions to CSV file
    :param filename:
    :param predictions:
    :return:
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Churn'])
        for pred in predictions:
            writer.writerow([pred])


def main() -> None:
    train_data = load_data(TRAINING_DATA_FILE_NAME)

    print(f"Training data rows: {len(train_data)}")

    train_labels = [row[0] for row in load_data(TRAINING_LABELS_FILE_NAME)]  # Removed [1:] to include first row if it's a label

    print(f"Training labels: {len(train_labels)}")

    val_data = load_data(VALIDATION_DATA_FILE_NAME)

    knn = KNNClassifier(k=5)

    X_train, y_train = knn.preprocess_data(train_data, train_labels)
    X_val = knn.preprocess_data(val_data)

    print(f"Processed training features: {len(X_train)}")
    print(f"Processed training labels: {len(y_train)}")

    knn.fit(X_train, y_train)

    val_predictions = knn.predict(X_val)

    save_predictions_to_csv('val_pred.csv', val_predictions)

    test_data = load_data('test.csv')
    X_test = knn.preprocess_data(test_data)
    test_predictions = knn.predict(X_test)
    save_predictions_to_csv('test_pred.csv', test_predictions)

if __name__ == '__main__':
    main()
