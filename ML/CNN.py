import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, roc_curve
import time
import matplotlib.pyplot as plt
import pandas as pd


class CNN_Classifier:
    """
    This class trains a 1D Convolutional Neural Network (CNN) model for classification tasks using group k-fold cross-validation.

    Attributes
    ----------
    features : numpy array
        Preprocessed feature set with shape (samples, time_steps, channels).
    target : numpy array
        Target variable for classification. Same length as the number of samples.
    groups : numpy array
        Indicating the chunk index for each sample.
    n_splits : int
        Number of folds for cross-validation.
    epochs : int
        Number of epochs for training the CNN.
    batch_size : int
        Batch size for training.
    """
    def __init__(self, features, target, groups, n_splits=5, epochs=20, batch_size=32):
        self.features = features
        self.target = target
        self.groups = groups
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size

    def create_cnn_model(self, input_shape):
        """
        Create a 1D Convolutional Neural Network (CNN) model using TensorFlow/Keras.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (time_steps, channels).

        Returns
        -------
        model : tf.keras.Model
            Compiled CNN model.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        """
        Train the CNN model using group k-fold cross-validation and evaluate using various metrics.

        Returns
        -------
        None
        """
        gkf = GroupKFold(n_splits=self.n_splits)
        results = []

        # To aggregate predicted and true targets for all folds
        y_true_list = []
        y_prob_list = []

        start_time = time.time()

        for train_index, test_index in gkf.split(self.features, self.target, self.groups):
            X_train, X_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.target[train_index], self.target[test_index]

            # Create a new CNN model for each fold
            model = self.create_cnn_model(input_shape=X_train.shape[1:])

            # Train the CNN model
            history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0,
                                validation_data=(X_test, y_test))


            # Make predictions
            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            y_prob = model.predict(X_test)

            y_true_list.extend(y_test)
            y_prob_list.extend(y_prob)

            # Evaluate metrics
            kappa = cohen_kappa_score(y_test, y_pred)
            mean_f1 = f1_score(y_test, y_pred, average='weighted')
            mean_sensitivity = recall_score(y_test, y_pred, average='macro')
            mean_precision = precision_score(y_test, y_pred, average='macro')
            mean_accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            # Store the results
            results.append({
                "Kappa": round(kappa, 3),
                "F1": round(mean_f1, 3),
                "Sensitivity": round(mean_sensitivity, 3),
                "Precision": round(mean_precision, 3),
                "Accuracy": round(mean_accuracy, 3),
                "AUC": round(auc, 3)
            })

        end_time = time.time()
        # Training time
        training_time = end_time - start_time
        print(f"training time: {training_time}")

        # Results as a DataFrame
        results_df = pd.DataFrame(results)
        print("\nFinal Cross-Validation Results:\n", results_df)

        # Calculating average results
        average_results = results_df.mean()
        print("\nAverage Cross-Validation Results:\n", average_results)

        # Calculating ROC-AUC
        fpr, tpr, _ = roc_curve(y_true_list, y_prob_list)
        plt.figure()
        plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title(f'ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

# TODO: -check all warnings