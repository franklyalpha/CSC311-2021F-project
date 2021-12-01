# TODO: complete this file.
from utils import *
import neural_network
import numpy as np
import torch
from torch.autograd import Variable
torch.manual_seed(0)

import pdb

def load_data(base_path="./data"):
    """Load the data.

    :param base_path: Path for loading the data. Defaults to "./data".
    :return: (train_matrix, valid_data, test_data)
        WHERE:
            train_matrix: 2D sparse matrix
            valid_data:  A dictionary {user_id: list,
            question_id: list, is_correct: list}
            test_data: A dictionary {user_id: list,
            question_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    train_data = load_train_csv(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    
    return train_matrix, train_data, valid_data, test_data

def bootstrap(train_matrix, seed):
    """Return a bootstrap sample of train_matrix. 

    :param train_matrix: 2D sparse matrix for training. 
    :param seed: Random seed for bootstrapping.
    :return: 2D sparse matrix.
    """
    rng = np.random.default_rng(seed)
    all_indices = [n for n in range(train_matrix.shape[0])]
    sample_indices = rng.choice(all_indices, size=len(all_indices)) 
    sample = train_matrix[sample_indices]
    return sample

def train(setting, train_matrix, train_data, valid_data, seed):
    """Train a neural network model with hyperparameters in setting on a 
    bootstrap sample of train_matrix, and return the trained model. 

    :param setting: Dictionary of hyperparameter settings for the model. 
    :param train_matrix: 2D sparse matrix provided for training (not boostrapped).
    :param valid_data: A dictionary {user_id: list,
                        question_id: list, is_correct: list}.
    :param seed: Random seed for bootstrapping.
    :return: Trained neural net model. 
    """
    # bootstrap train_matrix
    bootstrap_train = bootstrap(train_matrix, seed)
    # fill in missing entries with 0
    zero_bootstrap_train = bootstrap_train.copy()
    zero_bootstrap_train[np.isnan(zero_bootstrap_train)] = 0
    # change to Float Tensor for PyTorch
    bootstrap_train = torch.FloatTensor(bootstrap_train)
    zero_bootstrap_train = torch.FloatTensor(zero_bootstrap_train)
    
    # train model and generate predictions
    model = neural_network.AutoEncoder(bootstrap_train.shape[1], setting["k"])
    neural_network.train(model, setting["lr"], setting["lamb"], bootstrap_train, 
          zero_bootstrap_train, train_data, valid_data, setting["num_epoch"])
    return model

def bagging_train(settings, train_matrix, train_data, valid_data):
    """Train a neural network model for each of the hyperparameter setting 
    in settings on bootstrapped sample train_matrix. 

    :param settings: List[Dict] of model hyperparameter settings. 
                    Each dictionary has hyperparameter settings for one model. 
    :param train_matrix: 2D sparse matrix provided for training (not boostrapped).
    :param valid_data: A dictionary {user_id: list, 
                        question_id: list, is_correct: list}.
    :return: List of trained neural net models. 
    """
    models = []
    for count, setting in enumerate(settings):
        model = train(setting, train_matrix, train_data, valid_data, count)
        models.append(model)
    return models

def bagging_evaluate(models, train_matrix, valid_data):
    """For each trained neural net model in models, evaluate on valid_data and 
    get prediction accuracy. Average the predictions (0's and 1's) from each of
    the individual neural net models to get the predictions of the ensemble and
    also get the ensemble's prediction accuracy. 
    Print out the prediction accuracy of each of the individual neural net models
    and the ensemble. 

    :param models: List of trained neural net models. 
    :param train_matrix: 2D sparse matrix provided for training (not boostrapped).
    :valid_data: A dictionary {user_id: list,
                        question_id: list, is_correct: list}.
    """
    # fill in missing entries with 0
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # change to Float Tensor for PyTorch
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    
    predictions = []
    accuracies = []
    # Generate predictions on valid_data for each model and get prediction 
    # accuracy
    for model in models:
        model.eval()
        guesses = []
        for i, u in enumerate(valid_data["user_id"]):
            inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
            output = model(inputs)

            guess = output[0][valid_data["question_id"][i]].item() >= 0.5
            guesses.append(guess)   
        accuracy = get_accuracy(guesses, valid_data)
        predictions.append(guesses)
        accuracies.append(accuracy)
    
    # Average predictions of each model and compare the average with the 
    # threshold hold (0.5) to generate predictions for the ensemble
    sum_predictions = [sum(x) for x in zip(*predictions)]
    n = len(predictions)
    avg_predictions = [x/n for x in sum_predictions]
    ensemble_predictions = [
        (lambda x: 1 if x>=0.5 else 0)(x) for x in avg_predictions]
    ensemble_accuracy = get_accuracy(ensemble_predictions, valid_data)
    
    for i, acc in enumerate(accuracies):
        print(f"The accuracy of neural network model #{i+1} is: {acc}")
    print(f"The accuracy of the ensemble of neural network models is: \
        {ensemble_accuracy}")
    
    return None

def get_accuracy(predictions, valid_data):
    """Evaluate accuracy of predictions on valid_data. 

    :param predictions: List of predictions (0's and 1's) already generated 
                        on valid_data.
    :param valid_data: A dictionary {user_id: list, 
                        question_id: list, is_correct: list}. 
    """
    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        guess = predictions[i]
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct/total
   

def main():
    # Load data
    train_matrix, train_data, valid_data, test_data = load_data()
    
    # fill in missing entries with 0
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # change to Float Tensor for PyTorch
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    
    seeds = [0, 1, 2]
    settings = [{"k": 50, "lr": 0.01, "lamb": 0.001, "num_epoch": 40},
                {"k": 100, "lr": 0.01, "lamb": 0.001, "num_epoch": 50},
                {"k": 10, "lr": 0.01, "lamb": 0.01, "num_epoch": 30}]
    accuracies = []
    ##########################################################################
    # Code for training base model on each of the 3 bootstrap samples to
    # select the hyperparameter setting with the highest validation accuracy:
    ##########################################################################
    # for seed in seeds:
    #     for setting in settings:
    #         model = train(setting, train_matrix, train_data, valid_data, seed)
    #         valid_accuracy = neural_network.evaluate(model, zero_train_matrix, valid_data)
    #         accuracies.append(valid_accuracy)
    
    # i = 0
    # for seed in range(len(seeds)):
    #     for setting in range(len(settings)):
    #         print(f"Validation accuracy for sample {seed + 1} & setting {setting + 1}:\
    #             {accuracies[i]}")
    #         i += 1
    
    # We got that {"k": 50, "lr": 0.01, "lamb": 0.001, "num_epoch": 40} gives 
    # the best validation accuracy for all 3 bootstrap samples
    
    # Specify model settings
    settings = [{"k": 50, "lr": 0.01, "lamb": 0.001, "num_epoch": 40},
                {"k": 50, "lr": 0.01, "lamb": 0.001, "num_epoch": 40},
                {"k": 50, "lr": 0.01, "lamb": 0.001, "num_epoch": 40}]
    
    models = bagging_train(settings, train_matrix, train_data, valid_data)
    print("---Validation accuracy---")
    bagging_evaluate(models, train_matrix, valid_data)
    print("---Test accuracy---")
    bagging_evaluate(models, train_matrix, test_data)
    
if __name__ == "__main__":
    main()