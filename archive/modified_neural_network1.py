from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb

"""Corrupt known data in the sparse matrix and apply different weighting. 
"""

def load_data(base_path="./data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_data = load_train_csv(base_path)
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    return train_matrix, train_data, valid_data, test_data

def mask_data(train_matrix, percent):
    masked_train_matrix = np.copy(train_matrix)
    mask_indices_lst = []
    for u in range(train_matrix.shape[0]):
        # Find indices of not-null values in row u of sparse train_matrix
        user_row = train_matrix[u]
        # not_null_indices = user_row[np.argwhere(~np.isnan(user_row)).flatten()]
        not_null_indices = np.argwhere(~np.isnan(user_row)).flatten()
        rng = np.random.default_rng(u) # set seed to u
        mask_size = int(percent*len(not_null_indices))
        mask_indices = rng.choice(not_null_indices, size=mask_size, replace=False) 
        masked_train_matrix[u][mask_indices] = np.NAN
        mask_indices_lst.append(mask_indices)
        
    return masked_train_matrix, mask_indices_lst

def preprocess_data(train_matrix, masked_train_matrix):
    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    
    # Fill in the missing entries to 0.
    zero_masked_train_matrix = masked_train_matrix.copy()
    zero_masked_train_matrix[np.isnan(masked_train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_masked_train_matrix = torch.FloatTensor(zero_masked_train_matrix)

    train_matrix = torch.FloatTensor(train_matrix)
    return zero_train_matrix, zero_masked_train_matrix, train_matrix
     
    
class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        inner = self.g.forward(inputs)
        inner_activ = nn.Sigmoid()(inner)
        outer = self.h.forward(inner_activ)
        outer_activ = nn.Sigmoid()(outer)
        out = outer_activ
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_matrix, zero_train_matrix, 
          zero_masked_train_matrix, mask_indices_lst,
          train_data, valid_data, num_epoch, alpha, beta):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float -> for regularization
    :param train_matrix: 2D FloatTensor
    :param zero_train_matrix: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.
    # discuss possibilities: 1: adding the regularization term only once

    # Tell PyTorch you are training the model.
    model.train()
    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_matrix.shape[0]
    loss_recording = []
    valid_acc_record = []
    epoch_record = []
    for epoch in range(0, num_epoch):
        train_loss = 0.
        for user_id in range(num_student):
            inputs = Variable(zero_masked_train_matrix[user_id]).unsqueeze(0)
            target = Variable(zero_train_matrix[user_id]).unsqueeze(0)

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            norm = model.get_weight_norm()
            denom = 2
            mask_indices = mask_indices_lst[user_id]
            # not_mask_indices = torch.ones(output[0].numel(), dtype=torch.bool)
            # not_mask_indices[mask_indices] = False
            all_indices = np.array([x for x in range(train_matrix.shape[1])])
            not_mask_indices = np.setdiff1d(all_indices, mask_indices)

            loss = alpha*torch.sum(
                        (output[0][mask_indices] - target[0][mask_indices]) ** 2.) + \
                    beta*torch.sum(
                        (output[0][not_mask_indices] - target[0][not_mask_indices])**2.) + \
                    (lamb / denom) * norm

            # loss = torch.sum((output - target) ** 2.) + (lamb / denom) * norm
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        train_acc = evaluate(model, zero_train_matrix, train_data)
        valid_acc = evaluate(model, zero_train_matrix, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Train Acc: {}".format(epoch, train_loss, train_acc))
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        loss_recording.append(train_loss)
        valid_acc_record.append(valid_acc)
        epoch_record.append(epoch)
    plot(epoch_record, loss_recording, valid_acc_record)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def plot(epoches, losses, valid_accs):
    plt.clf()
    plt.title("training loss and epoch")
    plt.plot(epoches, losses)
    plt.legend()
    plt.savefig("network_reg_loss")
    plt.clf()
    plt.title("validation accuracy and epoch")
    plt.plot(epoches, valid_accs)
    plt.legend()
    plt.savefig("network_reg_accuracy")


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    train_matrix, train_data, valid_data, test_data = load_data()
    percent = 0.05
    masked_train_matrix, mask_indices_lst = mask_data(train_matrix, percent)
    # Change to float tensor for PyTorch
    zero_train_matrix, zero_masked_train_matrix, train_matrix = preprocess_data(
        train_matrix, masked_train_matrix)

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = 100
    model = AutoEncoder(train_matrix.shape[1], k)

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 20
    lamb = 0 #0.005
    alpha = 2
    beta = 1

    train(model, lr, lamb, train_matrix, zero_train_matrix, 
          zero_masked_train_matrix, mask_indices_lst,
          train_data, valid_data, num_epoch, alpha, beta)
    test_result = evaluate(model, zero_train_matrix, test_data)
    print("test accuracy: \n" + str(test_result))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
