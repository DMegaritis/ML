import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
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
    def __init__(self, features, target, groups, n_splits=5, epochs=20, batch_size=32, early_stopping=False):
        self.features = features
        self.target = target
        self.groups = groups
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping

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
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=256, kernel_size=5, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=512, kernel_size=5, activation='relu'),
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
        results_test = []
        results_train = []

        # To aggregate predicted and true targets for all folds (model evaluation on test data)
        y_true_list = []
        y_prob_list = []

        # Initialize lists to store true and predicted values (model evaluation on train data)
        y_train_true_list = []
        y_train_prob_list = []

        start_time = time.time()
        # Early Stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        for train_index, test_index in gkf.split(self.features, self.target, self.groups):
            X_train, X_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.target[train_index], self.target[test_index]

            # Create a new CNN model for each fold
            model = self.create_cnn_model(input_shape=X_train.shape[1:])

            # Train the CNN model
            history = model.fit(X_train, y_train,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                verbose=0,
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping] if self.early_stopping else [])

            # Make predictions
            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            y_prob = model.predict(X_test)

            # Lists with actual and predicted targets for the test data
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
            results_test.append({
                "Kappa": round(kappa, 3),
                "F1": round(mean_f1, 3),
                "Sensitivity": round(mean_sensitivity, 3),
                "Precision": round(mean_precision, 3),
                "Accuracy": round(mean_accuracy, 3),
                "AUC": round(auc, 3)
            })

            # Make predictions on the training set
            y_train_pred = (model.predict(X_train) > 0.5).astype("int32")
            y_train_prob = model.predict(X_train)

            # Aggregate true labels and predicted probabilities for the training set
            y_train_true_list.extend(y_train)
            y_train_prob_list.extend(y_train_prob)

            # Evaluating metrics for training set
            kappa_train = cohen_kappa_score(y_train, y_train_pred)
            mean_f1_train = f1_score(y_train, y_train_pred, average='weighted')
            mean_sensitivity_train = recall_score(y_train, y_train_pred, average='macro')
            mean_precision_train = precision_score(y_train, y_train_pred, average='macro')
            mean_accuracy_train = accuracy_score(y_train, y_train_pred)
            auc_train_table = roc_auc_score(y_train, y_train_prob)

            # Store the results for the training set
            results_train.append({
                "Kappa_train": round(kappa_train, 3),
                "F1_train": round(mean_f1_train, 3),
                "Sensitivity_train": round(mean_sensitivity_train, 3),
                "Precision_train": round(mean_precision_train, 3),
                "Accuracy_train": round(mean_accuracy_train, 3),
                "AUC_train": round(auc_train_table, 3)
            })

        end_time = time.time()
        # Training time
        training_time = end_time - start_time
        print(f"training time: {training_time}")

        # Results as a DataFrame (model evaluation on test data)
        results_test_per_fold = pd.DataFrame(results_test)
        print("\nPer Fold Cross-Validation Results for test set:\n", results_test_per_fold)
        # Saving the results from each fold to a CSV file
        table_path = r'C:\Users\klch3\PycharmProjects\MLTable\results/fold_wise_CNN.csv'
        results_test_per_fold.to_csv(table_path, mode='w')

        # Calculating average results (model evaluation on test data)
        average_results = round(results_test_per_fold.mean(), 3)
        print("\nAverage Cross-Validation Results for test set:\n", average_results)
        # Saving the aggregated results to a CSV file
        table_path = r'C:\Users\klch3\PycharmProjects\MLTable\results/aggregated_CNN.csv'
        average_results.to_csv(table_path, mode='w')

        # Results as a DataFrame (model evaluation on train data)
        results_train_per_fold = pd.DataFrame(results_train)
        print("\nPer fold Cross-Validation Results for train set:\n", results_train_per_fold)
        # Saving the results from each fold to a CSV file
        table_path = r'C:\Users\klch3\PycharmProjects\MLTable\results/fold_wise_train_CNN.csv'
        results_train_per_fold.to_csv(table_path, mode='w')

        # Calculating average results (model evaluation on test data)
        average_train_results = round(results_train_per_fold.mean(), 3)
        print("\nAverage Cross-Validation Results for train set:\n", average_train_results)
        # Saving the aggregated results to a CSV file
        table_path = r'C:\Users\klch3\PycharmProjects\MLTable\results/aggregated_train_CNN.csv'
        average_train_results.to_csv(table_path, mode='w')

        # Calculating ROC-AUC (model evaluation on test data)
        fpr_test, tpr_test, _ = roc_curve(y_true_list, y_prob_list)
        # AUC from the whole dataset
        # auc_test = roc_auc_score(y_true_list, y_prob_list)
        # AUC averaged over all folds
        auc_test = average_results.loc['AUC']
        plt.figure()
        plt.plot(fpr_test, tpr_test, color='orange', label=f'ROC curve (AUC = {auc_test:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title(f'ROC Curve for model evaluation')
        plt.legend(loc='lower right')
        # Saving the plot locally
        fig_path = r'C:\Users\klch3\PycharmProjects\MLTable\results\roc_curve_test.png'
        plt.savefig(fig_path, dpi=300)
        plt.show()

        # Calculating ROC-AUC (model evaluation on train data)
        fpr_train, tpr_train, _ = roc_curve(y_train_true_list, y_train_prob_list)
        # AUC from the whole dataset
        # auc_train = roc_auc_score(y_train_true_list, y_train_prob_list)
        # AUC averaged over all folds
        auc_train = average_train_results.loc['AUC_train']
        plt.figure()
        plt.plot(fpr_train, tpr_train, color='orange', label=f'ROC curve (AUC = {auc_train:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title(f'ROC Curve for training set')
        plt.legend(loc='lower right')
        # Saving the plot locally
        fig_path = r'C:\Users\klch3\PycharmProjects\MLTable\results\roc_curve_train.png'
        plt.savefig(fig_path, dpi=300)
        plt.show()

        # Ploting both ROC curves on the same figure
        plt.figure()
        plt.plot(fpr_test, tpr_test, color='orange', label=f'Test ROC (AUC = {auc_test:.2f})')
        plt.plot(fpr_train, tpr_train, color='blue', label=f'Train ROC (AUC = {auc_train:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve for Training and Test Sets')
        plt.legend(loc='lower right')

        # Save the figure
        fig_path = r'C:\Users\klch3\PycharmProjects\MLTable\results\roc_curve_train_vs_test.png'
        plt.savefig(fig_path, dpi=300)
        plt.show()

        # Trainning the final model on the entire dataset to save
        final_model = self.create_cnn_model(input_shape=self.features.shape[1:])

        history = final_model.fit(self.features, self.target,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  verbose=0)

        # Saving model
        final_model.save('final_cnn_classifier_model.h5')
        print("Final model saved as 'final_cnn_classifier_model.h5'")
