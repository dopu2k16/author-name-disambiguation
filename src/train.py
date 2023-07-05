from src.batch_score import batch_prediction
from src.models import get_ml_models
from src.preprocess import preprocess_dataset, load_data_csv

# Define the directory path containing the A++ files
# dir_path = '/path/to/folder/containing/xml/files'
dir_path = '/home/mitodru/Documents/interviews/springer-task/test-articles/'


def main():
    input_data = load_data_csv(dir_path+'authors.csv', delimiter=',')
    X_train, y_train, X_test, y_test = preprocess_dataset(input_data)
    # getting all the implemented ml models
    models = get_ml_models()
    # getting the predictions for both the training and testing datasets
    batch_prediction(X_train, y_train, X_test, y_test, models)


if __name__ == "__main__":
    main()
