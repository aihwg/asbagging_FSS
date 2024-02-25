from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    clustering_threshold: float #Threshold of hierarchical clustering(Default:0.95)
    param_leaf_rotation: int #Number of leaf_rotation of hierarchical clustering(Default:90)
    num_subsets: int #Number of subsets(Default:10)