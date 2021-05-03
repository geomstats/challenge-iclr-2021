import numpy as np
from sklearn import model_selection, metrics, svm, ensemble

model_params = {
    'svm': {
        'kernel': ['linear', 'rbf', 'sigmoid'], 
        'nu': [0.1, 0.3, 0.5]
    },
    
    'isoforest': {
        'n_estimators': [10, 100, 1000],
        'max_features': [0.1, 0.3, 1]
    },
}


def filter_dataset(receptor_feature, peptide_feature):
    """
    Processes features and removes None entries due to MDTraj/RDKit issues.

    Args:
        receptor_feature: list, arrays of receptor pdb features
        peptide_feature: list, arrays of peptide pdb features

    Returns:
        tuple(receptor_feature, peptide_feature, indices_list): lists of filtered pdb features and indices

    """
    peptide_indices_list = []
    receptor_indices_list = []
    
    for idx, feature in enumerate(peptide_feature):
        if feature is not None:
            peptide_indices_list.append(idx)
    
    for idx, feature in enumerate(receptor_feature):
        if feature is not None:
            receptor_indices_list.append(idx)
    
    indices_list = [idx for idx in peptide_indices_list if idx in receptor_indices_list]
    
    return (
        list(np.array(receptor_feature)[indices_list]), 
        list(np.array(peptide_feature)[indices_list]), 
        indices_list)

def occ_scorer(estimator, X, y=None):
    """Scores one-class classification.
    
    Args: 
        estimator: sklearn model, estimator from hyperparameter optimization.
        X: np array, test features.
        y: None, for consistency with sklearn.
        
    Returns:
        float, model accuracy
        
    """
    return sum([1 for idx in estimator.predict(X) if idx == 1])/len(X)

def occ_feature_preprocess(X, y=None, test_split=0.2, random_state=108):
    """Preprocesses features for sklearn model training.
    
    Args:
        X: list/np array, input features for model
        y: None, for consistency with sklearn
        test_split: float, split for testing, default=0.2
        random_state: int, seed - split for testing, default=108
        
    Returns:
        X_train: np array, input array for training
        y_train: np array, array of 1s with same length as X_train
        X_test: np array, input array for testing
        y_test: np array, array of 1s with same length as X_test
        
    """
    
    if type(X) is list:
        X = np.hstack([array for array in X])
        y = np.ones(len(X))
    
    return model_selection.train_test_split(
        X, y, test_size=test_split, random_state=random_state)

def occ_training(X_train, model_type, dict_params=None, val_split=0.25, random_state=108):
    """Trains one-class classifier by grid search.
    
    Args:
        X_train: np array, input for training
        model_type: str, type of model, example: svm, isoforest
        dict_params: dict, key: parameter, value: list of hyperparameters, default:model_params[model_type]
        val_split: float, validation split, default=0.25
        random_state: int, seed for splitting and isolation forest classifier
    
    Returns:
        best_model: sklearn model, best model
        best_params: dict, hyperparameters of best model
        best_accuracy: float, accuracy of best model
        
    """
    X_train, X_val, _, _ = model_selection.train_test_split(
        X_train, np.zeros(len(X_train)), 
        test_size=val_split, 
        random_state=random_state)
    
    if dict_params is None:
        dict_params = model_params[model_type]
        
    all_params = list(model_selection.ParameterGrid(dict_params))
    
    prev_accuracy = 0
    
    for tmp_params in all_params:
        if model_type is 'svm':
            tmp_model = svm.OneClassSVM(cache_size=5000)
            tmp_model.set_params(kernel=tmp_params['kernel'])
            tmp_model.set_params(nu=tmp_params['nu'])
        elif model_type is 'isoforest':
            tmp_model = ensemble.IsolationForest(
                n_jobs=-1, warm_start=True, random_state=random_state)
            tmp_model.set_params(n_estimators=tmp_params['n_estimators'])
            tmp_model.set_params(max_features=tmp_params['max_features'])
        
        tmp_model.fit(X_train)
        val_accuracy = occ_scorer(tmp_model, X_val)
        
        if val_accuracy > prev_accuracy:
            best_model = tmp_model
            best_params = tmp_params
            best_accuracy = val_accuracy
    
    return best_model, best_params, best_accuracy