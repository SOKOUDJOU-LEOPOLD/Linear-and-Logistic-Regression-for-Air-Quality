import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from typing import Tuple, List

'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
the library above available.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional 
default arguments, class parameters, or methods to help you complete the assignment.

3. Some initial steps or definition are given, aiming to help you getting started. As long as you follow 
the argument and return type, you are free to change them as you see fit.

4. Your code should be free of compilation errors. Compilation errors will result in 0 marks.
'''

class DataProcessor:
    def __init__(self, data_root: str):
        """Initialize data processor with paths to train and test data.
        
        Args:
            data_root: root path to data directory
        """
        self.data_root = data_root
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data from CSV files.
        
        Returns:
            Tuple containing training and test dataframes
        """
        # TODO: Implement data loading
        train_data_file_name = "hw1_data_train.csv"
        test_data_file_name = "hw1_data_test.csv"
        train_data = pd.read_csv(f"{self.data_root}/{train_data_file_name}")
        test_data = pd.read_csv(f"{self.data_root}/{test_data_file_name}")
        
        return (train_data, test_data)
        
        
        
    def check_missing_values(self, data: pd.DataFrame) -> int:
        """Count number of missing values in dataset.
        
        Args:
            data: Input dataframe
            
        Returns:
            Number of missing values
        """
        # TODO: Implement missing value check
        return data.isnull().sum().sum()
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values.
        
        Args:
            data: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        # TODO: Implement data cleaning
        # 1. Replace sensor missing flag -200
        data = data.replace(-200, np.nan)
        # drop the missing data
        return data.dropna()
        
    def extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataframe, convert to numpy arrays.
        
        Args:
            data: Input dataframe
            
        Returns:
            Tuple of feature matrix X and label vector y
        """
        # TODO: Implement feature/label extraction
        # feature matrix will be X and Label Vector will be Y
        X = data.drop(columns=["PT08.S1(CO)"])
        Y = data["PT08.S1(CO)"]

        return (X, Y)
    
    # Normalize Data
    def normalize(self, X_train, X_test):
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        return X_train, X_test
    

class ExploratoryDataAnalysis:
    def plotHistograms(self, data:pd.DataFrame):
        plt.figure(figsize=(15, 8))
        for i, col in enumerate(data.columns):
            plt.subplot(3,4,i+1)
            plt.hist(data[col], bins = 25)
            plt.title(col)
            plt.xlabel("Value")
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    # Choose T and AH
    def scatterPlot(self, data: pd.DataFrame, x_feature:str, y_feature: str):
        plt.figure(figsize=(5,5))
        plt.scatter(data[x_feature],data[y_feature], alpha = 0.5)
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f"Scatter Plot of {x_feature} VS {y_feature}")
        plt.show()

    def plotCorrelationHeatmap(self, data: pd.DataFrame):
        corr_matrix = data.corr(method="pearson")
        
        plt.figure(figsize=(11, 9))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True
        )
        plt.title("Pearson Correlation Heatmap")
        plt.show()
        
        return corr_matrix


class LinearRegression:
    def __init__(self, learning_rate: float = 0.001, max_iter: int = 1000):
        """Initialize linear regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train linear regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement linear regression training
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # store losse for each iteration
        losses = []
        
        # training
        for i in range(self.max_iter):
            # predicted value 
            y_pred = X @ self.weights + self.bias
            
            # loss value
            loss = self.criterion(y, y_pred)
            losses.append(loss)
            
            # gradients
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return losses        

    def plotLoss(self, losses:np.ndarray):
        plt.plot(losses)
        plt.xlabel("Iterations")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss vs Iterations")
        plt.show()

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement linear regression prediction
        return X @ self.weights + self.bias

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MSE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
        return np.mean((y_true - y_pred) ** 2)

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Metric value
        """
        # TODO: Implement RMSE calculation
        return np.sqrt(self.criterion(y_true, y_pred))
    
    def tuning_loop(learning_rates, iterations_list):            
        best_rmse = float("inf")
        best_model = None
        best_loss = None

        for lr in learning_rates:
            for iters in iterations_list:

                print(f"\nTraining with lr={lr}, iterations={iters}")

                model = LinearRegression(learning_rate=lr, max_iter=iters)

                losses = model.fit(X_train_norm, Y_train)

                preds = model.predict(X_train_norm)

                rmse = np.sqrt(np.mean((Y_train - preds) ** 2))

                print("RMSE:", rmse)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_loss = losses
                
                # stop tuning when we reach rmse <= 71
                if rmse <= 71:
                    return (best_rmse, best_model, best_loss)
        
        return (best_rmse, best_model, best_loss)

def main_LinearRegression():
    # ===============================
    # Step 1 — Load data
    # ===============================
    processor = DataProcessor("./data")   # change path if needed
    train_df, test_df = processor.load_data()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)


    # ===============================
    # Step 2 — Check missing values
    # ===============================
    missing_train = processor.check_missing_values(train_df)
    missing_test = processor.check_missing_values(test_df)

    print("Missing values (train):", missing_train)
    print("Missing values (test):", missing_test)


    # ===============================
    # Step 3 — Clean data
    # ===============================
    train_df = processor.clean_data(train_df)
    test_df = processor.clean_data(test_df)


    # ===============================
    # Step 4 — Extract features/labels
    # ===============================
    X_train, y_train = processor.extract_features_labels(train_df)
    X_test, y_test = processor.extract_features_labels(test_df)


    # ===============================
    # Step 5 — Normalize features (VERY IMPORTANT)
    # ===============================
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std   # use TRAIN stats only


    # ===============================
    # Step 6 — Hyperparameter tuning loop
    # ===============================
    learning_rates = [0.0005, 0.001, 0.005]
    iterations_list = [1000, 3000, 5000]

    best_rmse = float("inf")
    best_model = None

    for lr in learning_rates:
        for iters in iterations_list:

            print(f"\nTraining with lr={lr}, iters={iters}")

            model = LinearRegression(learning_rate=lr, max_iter=iters)
            losses = model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = model.metric(y_test, preds)

            print("RMSE:", rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_losses = losses


    print("\n===============================")
    print("BEST RMSE:", best_rmse)
    print("===============================")


    # ===============================
    # Step 7 — Plot final loss curve
    # ===============================
    plt.figure(figsize=(6, 4))
    plt.plot(best_losses)
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.show()


    # ===============================
    # Step 8 — Final predictions
    # ===============================
    final_preds = best_model.predict(X_test)
    print("First 10 predictions:", final_preds[:10])


class LogisticRegression:
    def __init__(self):
        """Initialize logistic regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
        """
        self.weights = None
        self.bias = None
        self.learning_rate = 0.001
        self.max_iter = 1000
        self.l2_lambda = 0.0

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = (X - self.mean) / self.std

        y = self.label_binarize(y)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        losses = []

        for _ in range(self.max_iter):

            # Forward
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # Loss
            loss = self.criterion(y, y_pred)
            losses.append(loss)

            # Gradients
            dw = (X.T @ (y_pred - y)) / n_samples + self.l2_lambda * self.weights
            db = np.mean(y_pred - y)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return losses
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction probabilities using normalized features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities
        X = (X - self.mean) / self.std
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    # Loss Function
    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate BCE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
        eps = 1e-9
        return -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )
    
    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score with handling of edge cases.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            F1 score (between 0 and 1), or 0.0 for edge cases
        """
        # TODO: Implement F1 score calculation

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        if tp == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return 2 * precision * recall / (precision + recall)

    def label_binarize(self, y: np.ndarray) -> np.ndarray:
        """Binarize labels for binary classification.
        
        Args:
            y: Target vector
            
        Returns:
            Binarized labels
        """
        # TODO: Implement label binarization
        return (y > 1000).astype(int)

    def get_auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            AUROC score (between 0 and 1)
        """
        # TODO: Implement AUROC calculation
        thresholds = np.linspace(0, 1, 200)
        tpr_list = []
        fpr_list = []

        for t in thresholds:
            preds = (y_pred >= t).astype(int)

            tp = np.sum((y_true == 1) & (preds == 1))
            fp = np.sum((y_true == 0) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            tn = np.sum((y_true == 0) & (preds == 0))

            tpr = tp / (tp + fn + 1e-9)
            fpr = fp / (fp + tn + 1e-9)

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # trapezoidal rule
        order = np.argsort(fpr_list)
        return np.trapz(np.array(tpr_list)[order], np.array(fpr_list)[order])


    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation
        y_true = self.label_binarize(y_true)
        return self.get_auroc(y_true, y_pred)

    def plotLoss(self, losses:np.ndarray):
        plt.plot(losses)
        plt.xlabel("Iterations")
        plt.ylabel("BCE Loss")
        plt.title("Training Loss vs Iterations")
        plt.show()

class ModelEvaluator:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize evaluator with number of CV splits.
        
        Args:
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
    def cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform cross-validation
        
        Args:
            model: Model to be evaluated
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of metric scores
        """
        # TODO: Implement cross-validation

if __name__ == "__main__":
    
    # 3.1  Data Processing
    
    # load data
    dataProcessor = DataProcessor("./")
    train_data, test_data = dataProcessor.load_data()
    print("Head of train data: \n", train_data.head())
    print("Shape of train data: ", train_data.shape)
    print("Head of test data: \n", test_data.head())
    print("Shape of test data: ", test_data.shape)
    print()

    # Missing data
    train_missing_data = dataProcessor.check_missing_values(train_data)
    test_missing_data = dataProcessor.check_missing_values(test_data)
    print("Number of Missing data in train data: ", train_missing_data)
    print("Number of Missing data in test data: ", test_missing_data)
    print()

    # drop missing data
    train_clean_data = dataProcessor.clean_data(train_data)
    test_clean_data = dataProcessor.clean_data(test_data)
    print("Shape of train clean data: ", train_clean_data.shape)
    print("Shape of test clean data: ", test_clean_data.shape)
    print()
    
    # Extract Features and label for tain data but Features only for test data since it does not have a label
    X_train, Y_train = dataProcessor.extract_features_labels(train_clean_data)
    X_test = test_clean_data
    print("train data features and label: ")
    print("features: \n", X_train)
    print("label: \n", Y_train)

    #Normalize data
    X_train_norm, X_test_norm = dataProcessor.normalize(X_train, X_test)

    # 3.2 Exploratory Data Analysis

    # Plot the histograms of all the features in the data 
    # ExploratoryDataAnalysis().plotHistograms(train_clean_data)
    # get the mean, std, min, max for each features. Describing clean data
    print("Features Description:\n", train_clean_data.describe())

    # Picked T and AH and create a scatter plot to illustrate the correlation between them
    # ExploratoryDataAnalysis().scatterPlot(test_clean_data, "T", "RH")

    # Compute the Pearson’s correlation between all pairs of variables 1-12
    # ExploratoryDataAnalysis().plotCorrelationHeatmap(train_clean_data)

    # 3.3 Linear Regression

    # convert pd.DataFrames to numpy.Array for math operations
    X_train_norm = X_train_norm.values
    Y_train = Y_train.values
    X_test_norm = X_test_norm.values

    # Construct main training loop without tuning hyperparameters, record loss, plot loss against iteration 
    linear_model = LinearRegression()
    # losses = linear_model.fit(X_train_norm, Y_train)
    # linear_model.plotLoss(losses)

    # Make prediction using trained model
    # test_predictions = linear_model.predict(X_test_norm)
    # print("\nUnTuned Test predictions shape:", test_predictions.shape)

    # save the predictions of the untuned trained model
    # np.savetxt("linear_predictions_untuned.csv", test_predictions, delimiter=",")

    # Tune Hyperparameters to achieve RMSE <= 71
    # define hyperparameters tuning candidates
    learning_rates = [0.001, 0.01]
    iterations_list = [1000, 2000]
    # best_rmse, best_model, best_loss = LinearRegression.tuning_loop(learning_rates, iterations_list)
    # print("Best Model: lr = ", best_model.learning_rate, ", iterations = ", best_model.max_iter)

    # Plot training loss for tuned trained model
    # best_model.plotLoss(best_loss)

    # Make prediction using tuned trained model
    # test_predictions_tuned = best_model.predict(X_test_norm)
    # print("\nTuned Test predictions shape:", test_predictions_tuned.shape)

    # save the predictions of the tuned trained model
    # np.savetxt("linear_predictions_tuned.csv", test_predictions, delimiter=",")


    # 3.4 Logistic Regression
    print("|                                                    |")
    print("================= Logistic Regression ================")
    X_train_log, Y_train_log = dataProcessor.extract_features_labels(train_clean_data)
    X_test_log = test_clean_data
    print("train data features and label: ")
    print("features: \n", X_train_log)
    print("label: \n", Y_train_log)

    # test set has NO labels → only features
    X_test_log = test_clean_data

    # Normalize (fit only on train stats)
    X_train_log_norm, X_test_log_norm = dataProcessor.normalize(
        X_train_log, X_test_log
    )

    # convert pd.DataFrames to numpy.Array for math operations
    X_train_log_norm = X_train_log_norm.values
    Y_train_log = Y_train_log.values
    X_test_log_norm = X_test_log_norm.values

    # Construct main training loop, record loss, and plot the loss against iterations (Untuned model)
    log_model = LogisticRegression()

    losses = log_model.fit(X_train_log_norm, Y_train_log)

    log_model.plotLoss(losses)

    test_pred = log_model.predict(X_test_log_norm)

    np.savetxt("logistic_predictions_untuned.csv", test_pred, delimiter=",")


    




