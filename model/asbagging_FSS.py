import numpy as np 

from model.classification_functions import classification_functions
from model.config import Config

class asbagging_FSS():
    """
    Attributes:
        bootstrap (bool): Whether to use bootstrapping.
        models (list): List of trained SVM models.
        selected_features (list): List of selected feature indices.
    """
    def __init__(self,config:Config,bootstrap=True):
        """
        Initialize ASBagging_FSS.

        Parameters:
            bootstrap (bool): Whether to use bootstrap sampling (default is True).
        """
        super().__init__()
        self.config=config
        self.classification_functions=classification_functions(config)
        self.bootstrap=bootstrap
        self.models=None
        self.selected_features=None

    def fit(self, my_data, labels):
        """
        Fit the model to the training data.

        Args:
            my_data (ndarray): Input data with shape (n_samples, n_features).
            labels (ndarray): Labels for the data.
        """
        cluster_ids=self.classification_functions.hierarchical_peason(my_data)#各特徴のクラスタ番号
        self.selected_features = self.classification_functions.select_feature_by_snr_for_each_cluster(my_data, cluster_ids, labels)
        subsets = self.classification_functions.create_subsets(my_data, labels, self.selected_features, self.config.num_subsets)
        self.models = self.classification_functions.train_svm_on_subsets(subsets)

    def predict(self, vector):
        """
        Make predictions on new data.

        Args:
            vector (ndarray): New data with shape (n_samples, n_features).

        Returns:
            ndarray: Predicted labels for the new data.
        """
        pred=np.array([],dtype=int)
        for i in range(vector.shape[0]):
            predictions = []
            for model, index_features in zip(self.models, self.selected_features):
                sampled_vector = vector[i, index_features]
                predictions.append(model.predict(sampled_vector.reshape(-1,1)))
            pred=np.append(pred,max(predictions, key=predictions.count))
        return pred
