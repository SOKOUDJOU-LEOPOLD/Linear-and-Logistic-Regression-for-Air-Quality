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
        
        print(train_data.head())
        print(train_data.shape())

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

class ExploratoryDataAnalysis:
    def plotHistograms(data:pd.DataFrame):
        plt.figure(figsize=(15, 20))
        for i, col in enumerate(data.columns):
            plt.subplot(3,4,i+1)
            plt.hist(data[col], bins = 25)
            plt.title(col)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()

    # Choose T and AH
    def scatterPlot(data: pd.DataFrame, x_feature:str, y_feature: str):
        plt.figure(figsize=(5,5))
        plt.scatter(data[x_feature],data[y_feature], alpha = 0.5)
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f"Scatter Plot of {x_feature} VS {y_feature}")
        plt.show()

    def plotCorrelationHeatmap(data: pd.DataFrame):
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
    def __init__(self):
        """Initialize linear regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """
        self.weights = None
        self.bias = None
        self.learning_rate = 0.001
        self.max_iter = 1000
        
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

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Metric value
        """
        # TODO: Implement RMSE calculation

class LogisticRegression:
    def __init__(self):
        """Initialize logistic regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
        """
        self.weights = None
        self.bias = None
        self.learning_rate = None
        self.max_iter = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction probabilities using normalized features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate BCE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
    
    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score with handling of edge cases.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            F1 score (between 0 and 1), or 0.0 for edge cases
        """
        # TODO: Implement F1 score calculation

    def label_binarize(self, y: np.ndarray) -> np.ndarray:
        """Binarize labels for binary classification.
        
        Args:
            y: Target vector
            
        Returns:
            Binarized labels
        """
        # TODO: Implement label binarization

    def get_auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            AUROC score (between 0 and 1)
        """
        # TODO: Implement AUROC calculation

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation

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
    print("Hello World!")