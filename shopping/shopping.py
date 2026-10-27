import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            administrative = int(row["Administrative"])
            administrative_duration = float(
                row["Administrative_Duration"]
            )
            informational = int(row["Informational"])
            informational_duration = float(
                row["Informational_Duration"]
            )
            product_related = int(row["ProductRelated"])
            product_related_duration = float(
                row["ProductRelated_Duration"]
            )
            bounce_rates = float(row["BounceRates"])
            exit_rates = float(row["ExitRates"])
            page_values = float(row["PageValues"])
            special_day = float(row["SpecialDay"])

            month_names = [
                "Jan", "Feb", "Mar", "Apr", "May", "June",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ]
            month = month_names.index(row["Month"])

            operating_systems = int(row["OperatingSystems"])
            browser = int(row["Browser"])
            region = int(row["Region"])
            traffic_type = int(row["TrafficType"])

            visitor_type = (
                1 if row["VisitorType"] == "Returning_Visitor"
                else 0
            )

            weekend = 1 if row["Weekend"] == "TRUE" else 0

            evidence.append([
                administrative,
                administrative_duration,
                informational,
                informational_duration,
                product_related,
                product_related_duration,
                bounce_rates,
                exit_rates,
                page_values,
                special_day,
                month,
                operating_systems,
                browser,
                region,
                traffic_type,
                visitor_type,
                weekend
            ])

            labels.append(
                1 if row["Revenue"] == "TRUE" else 0
            )

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            if predicted == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if predicted == 0:
                true_negative += 1
            else:
                false_positive += 1

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0
    )

    return sensitivity, specificity


if __name__ == "__main__":
    main()
