from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import transform_data


def get_ml_models():
    """
    The ML training algorithms for the Lead Generator Problem.
    The following ml algorithms were used to predict a given customer as a hot lead or not.
    """
    models = dict()
    vect, transformer = transform_data()
    # Logistic regression
    model = LogisticRegression()
    models['LR'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Multinomial Naive Bayes
    model = MultinomialNB(alpha=.01)
    models['MultiNaiveBayes'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Multinomial Naive Bayes
    model = model = LogisticRegression()
    # define the ovr strategy
    model = OneVsRestClassifier(LinearSVC(random_state=0, tol=1e-5, multi_class="ovr", class_weight='balanced'))
    models['MultiNaiveBayes'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    model = OneVsRestClassifier(LinearSVC(random_state=0, tol=1e-5, multi_class="ovr", class_weight='balanced'))
    models['SVM'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Perceptron
    model = Perceptron()
    models['Perceptron'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Decision Tree
    model = DecisionTreeClassifier(random_state=122, class_weight="balanced")
    models['CART'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Random Forest
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    models['RandomForest'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Gradient Boosting
    model = GradientBoostingClassifier()
    models['GBM'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Multilayer Perceptron
    model = MLPClassifier(random_state=1, early_stopping=True)
    models['MLP'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    return models
