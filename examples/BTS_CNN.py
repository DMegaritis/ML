from ML.CNN import CNN_Classifier
import numpy as np

# Specifying parameters
file_path = r'C:\Users\klch3\PycharmProjects\MLTable\data/cleaned_aggregated_BTS24.npy'
features = np.load(file_path)
file_path = r'C:\Users\klch3\PycharmProjects\MLTable\data/target_BTS24.npy'
target = np.load(file_path)
groups = np.arange(features.shape[0])

# Creating an instance of CNN_Classifier
cnn_classifier = CNN_Classifier(features=features, target=target, groups=groups, n_splits=5, epochs=20, batch_size=32)
cnn_classifier.train()

