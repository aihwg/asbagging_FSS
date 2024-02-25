import random
import pandas as pd
import numpy as np 
from sklearn import svm
from sklearn.utils import resample
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import tstd

from model.config import Config

class classification_functions():
    def __init__(self,config:Config):
        super().__init__()
        self.config=config

    def hierarchical_peason(self,a):
        """
        Perform hierarchical clustering based on Pearson correlation.

        Args:
            a (list or ndarray): Input data with shape (n_samples, n_features).

        Returns:
            ndarray: Cluster IDs for each feature.
        """
        a = pd.DataFrame(a)
        correlations=a.corr(method='pearson')
        dissimilarity = 1 - abs(correlations)
        Z = linkage(squareform(dissimilarity), 'complete')
        dendrogram(Z, labels=a.columns, orientation='top', leaf_rotation=self.config.param_leaf_rotation)
        cluster_ids_x = fcluster(Z, self.config.clustering_threshold, criterion='distance')
        return cluster_ids_x

    def calculate_snr(self,data, feature_idx, labels):
        """
        Calculate Signal-to-Noise Ratio (SNR) for a specific feature.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).
            feature_idx (int): Index of the feature.
            labels (ndarray): Labels for the data.

        Returns:
            float: SNR value.
        """
        class0 = data[labels == 0, feature_idx]
        class1 = data[labels == 1, feature_idx]
        if len(class0) > 1:
            mu0 = class0.mean()
            sigma0 = tstd(class0)
        else:
            mu0 = 0
            sigma0 = 1e-10  # small value to prevent division by zero
        if len(class1) > 1:
            mu1 = class1.mean()
            sigma1 = tstd(class1)
        else:
            mu1 = 0
            sigma1 = 1e-10  # small value to prevent division by zero
        snr = abs(mu0 - mu1) / (sigma0 + sigma1)
        return snr

    def select_feature_by_snr(self,data, labels):
        """
        Select the feature with the highest SNR.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).
            labels (ndarray): Labels for the data.

        Returns:
            int: Index of the selected feature.
        """
        num_features = data.shape[1]
        snr_scores = [self.calculate_snr(data, i, labels) for i in range(num_features)]
        return np.argmax(snr_scores)

    def select_feature_by_snr_for_each_cluster(self,data, labels, cluster_ids):
        """
        Select the feature with the highest SNR for each cluster.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).
            labels (ndarray): Labels for the data.
            cluster_ids (ndarray): Cluster IDs for each feature.

        Returns:
            list: List of selected feature indices for each cluster.
        """
        unique_clusters = np.unique(cluster_ids)
        selected_features = []

        for cluster in unique_clusters:
            cluster_indices = np.where(cluster_ids == cluster)[0]
            cluster_data = data[cluster_indices]
            cluster_labels = labels[cluster_indices]
            selected_feature = self.select_feature_by_snr(cluster_data, cluster_labels)
            selected_features.append(selected_feature)

        return selected_features

    def create_subsets(self,data, labels, selected_features, num_subsets):
        """
        Create subsets of data for each selected feature.

        Args:
            data (ndarray): Input data with shape (n_samples, n_features).
            labels (ndarray): Labels for the data.
            selected_features (list): List of selected feature indices.
            num_subsets (int): Number of subsets to create.

        Returns:
            list: List of tuples containing downsampled data and labels for each subset.
        """
        subsets = []
        for _ in range(num_subsets):
            feature_idx = random.choice(selected_features)
            subset_data = data[:, feature_idx]
            
            # Determine minority and majority classes based on label counts
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            minority_label = unique_labels[np.argmin(label_counts)]
            majority_label = unique_labels[np.argmax(label_counts)]
            
            # Separate minority and majority classes
            minority_class = subset_data[labels == minority_label]
            majority_class = subset_data[labels == majority_label]
            
            # Check if minority and majority classes are equal in size
            if len(minority_class) == len(majority_class):
                downsampled_data = subset_data
                downsampled_labels = labels
            else:
                # Upsample minority class
                majority_downsampled = resample(majority_class,
                                            replace=True,  # sample with replacement
                                            n_samples=len(minority_class),  # to match majority class
                                            random_state=123)  # reproducible results
                
                # Combine majority class with upsampled minority class
                downsampled_data = np.hstack((minority_class, majority_downsampled))
                downsampled_labels = np.hstack((np.full(len(minority_class), minority_label),
                                            np.full(len(majority_downsampled), majority_label)))
            
            # Append the new subset to the list
            subsets.append((downsampled_data, downsampled_labels))
            
        return subsets

    def train_svm_on_subsets(self,subsets):
        """
        Train SVM models on subsets of data.

        Args:
            subsets (list): List of tuples containing downsampled data and labels for each subset.

        Returns:
            list: List of trained SVM models.
        """
        models = []
        for subset_data, subset_labels in subsets:
            model = svm.SVC()
            # model=ExtraTreeClassifier()
            model.fit(subset_data.reshape(-1, 1), subset_labels)
            models.append(model)
        return models

    def predict_with_models(self,models, test_data, selected_features):
        """
        Make predictions using trained models.

        Args:
            models (list): List of trained SVM models.
            test_data (ndarray): Test data with shape (n_samples, n_features).
            selected_features (list): List of selected feature indices.

        Returns:
            ndarray: Predicted labels for the test data.
        """
        predictions = []
        for model, feature_idx in zip(models, selected_features):
            feature_data = test_data[:, feature_idx]
            prediction = model.predict(feature_data.reshape(-1, 1))
            predictions.append(prediction)
        
        # 多数決を行い、最終的な予測を返す
        final_prediction = np.round(np.mean(predictions, axis=0))  # 多数決を行う
        return final_prediction
        