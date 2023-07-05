import csv
import os
import pickle

from matplotlib import pyplot as plt
from numpy import mean, std
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

from src.evaluate import evaluate_model_predict


def batch_prediction(X_train, y_train, X_test, y_test, models):
    """
    Finding the predictions for each model on the dataset by calling the evaluate_model()
    """
    #  the models and store results
    results, names = list(), list()
    # test prediction list for all models
    test_preds = []
    # directory for storing the test predictions and classifier models
    pred_dir = './results'
    model_dir = './models'
    # creating the results directory
    try:
        os.makedirs(pred_dir)
        print("folder '{}' created ".format(pred_dir))
    except FileExistsError:
        print("folder {} already exists".format(pred_dir))
    # creating the results directory
    try:
        os.makedirs(model_dir)
        print("folder '{}' created ".format(pred_dir))
    except FileExistsError:
        print("folder {} already exists".format(pred_dir))

    for name, model in models.items():
        # prediction and evaluation scores for each model
        scores, y_test_pred, test_acc, test_p, test_r, test_f1 = evaluate_model_predict(model, X_train, y_train,
                                                                                        X_test, y_test)
        # appending the test prediction of the respective model
        test_preds.append((model, y_test_pred))
        print("Saving predictions")

        # writing the ground truth label and test prediction into file
        with open(pred_dir + "/" + 'pred_' + f'{name}.txt', "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(y_test, y_test_pred))

        # appending the evaluation scores
        results.append(scores)
        # appending the names of the model
        names.append(name)
        # printing the training accuracy
        print('Average Training accuracy for %s is %.3f (%.3f)'
              % (name, mean(scores['train_accuracy']) * 100,
                 std(scores['train_accuracy'])))

        # printing the validation accuracy
        print('Average Validation accuracy for %s is %.3f (%.3f)'
              % (name, mean(scores['test_accuracy']) * 100,
                 std(scores['test_accuracy'])))
        # printing thr training F1-score
        print('Average Training F1 score for %s is %.3f (%.3f)'
              % (name, mean(scores['train_f1_weighted']),
                 std(scores['train_f1_weighted'])))
        # printing the validation F1-score
        print('Average Validation F1-score for %s is %.3f (%.3f)'
              % (name, mean(scores['test_f1_weighted']),
                 std(scores['test_f1_weighted'])))
        # printing the test accuracy of a model
        print('Test accuracy for %s is %.3f' % (name, test_acc * 100))
        # printing the test precision of a model
        print('Test Precision for %s is %f' % (name, test_p))
        # printing the test recall of a model
        print('Test Recall for %s is %.3f' % (name, test_r))
        # printing the test F1-score of a model
        print('Test F1-score for %s is %.3f' % (name, test_f1))
        print(metrics.classification_report(y_test, y_test_pred))
        cm = metrics.confusion_matrix(y_test, y_test_pred)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.show()
        print('-------------------------------------------------')

        # save the model to disk
        filename = f'./models/{name}.pkl'
        pickle.dump(model, open(filename, 'wb'))
        # saving the input features of the model
        model_columns = list(X_train.columns)
        with open(f'./models/{name}_inp_features.pkl', 'wb') as file:
            pickle.dump(model_columns, file)
