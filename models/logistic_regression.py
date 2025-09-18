from sklearn.linear_model import LogisticRegression

def get_model():
    """Return Logistic Regression model"""
    return LogisticRegression(max_iter=1000)
