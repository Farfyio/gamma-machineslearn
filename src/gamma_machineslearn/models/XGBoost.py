from xgboost import XGBClassifier

def xgboost_factory():
    """
    Factory function to create an instance of XGBClassifier with default parameters.
    
    Returns:
        XGBClassifier: An instance of the XGBClassifier.
    """
    return XGBClassifier()