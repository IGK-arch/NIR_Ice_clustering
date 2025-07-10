from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_and_evaluate_classifier(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)
    return clf, report
