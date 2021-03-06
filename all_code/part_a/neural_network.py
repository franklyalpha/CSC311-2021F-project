from all_code.utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data(base_path="../data"):
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
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    train_data = load_train_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, train_data, valid_data, test_data


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


def train(model, lr, lamb, train_matrix, zero_train_matrix, train_data, valid_data, num_epoch):
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
    highest_valid_acc = 0
    for epoch in range(0, num_epoch):
        train_loss = 0.
        # norm = model.get_weight_norm()
        # denom = 2
        for user_id in range(num_student):

            inputs = Variable(zero_train_matrix[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            norm = model.get_weight_norm()
            denom = 2
            loss = torch.sum((output - target) ** 2.) + (lamb / denom) * norm
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        train_acc = evaluate(model, zero_train_matrix, train_data)
        valid_acc = evaluate(model, zero_train_matrix, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Train Acc: {}".format(epoch, train_loss, train_acc))
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        loss_recording.append(train_acc)
        valid_acc_record.append(valid_acc)
        epoch_record.append(epoch)
        highest_valid_acc = max(highest_valid_acc, valid_acc)

    # with open("nn0_valid_acc.txt", "w") as fp:
    #     json.dump(valid_acc_record, fp)

    plot(epoch_record, loss_recording, valid_acc_record)
    return highest_valid_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def plot(epoches, losses, valid_accs):
    plt.clf()
    plt.title("training accuracy, validation accuracy and epoch")
    plt.plot(epoches, valid_accs)
    plt.plot(epoches, losses)
    plt.legend(["validation accuracy", "training accuracy"])
    plt.savefig("network_reg_loss")



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
    zero_train_matrix, train_matrix, train_data, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    # k_list = [10, 50, 100, 200, 500]
    # k_accuracy = []
    # for k in k_list:
    #     model = AutoEncoder(train_matrix.shape[1], k)
    #
    # # Set optimization hyperparameters.
    #     lr = 0.01
    #     lr = 0.001
    #     lr = 0.1
    #     num_epoch = 200
    #     lamb = 0.1
    #
    #     highest_acc = train(model, lr, lamb, train_matrix, zero_train_matrix,
    #         valid_data, num_epoch)
    #     test_result = evaluate(model, zero_train_matrix, test_data)
    #     print("test accuracy: \n" + str(test_result))
    #     k_accuracy.append(highest_acc)
    # best_k = k_accuracy.index(max(k_accuracy))
    # print("will use this k for further procedure due to its highest accuracy:" + str(best_k))

    # lambda_list = [1, 0.1, 0.01, 0.001]
    # lambda_accuracy = []
    # for lamb in lambda_list:
    #     k = 50
    #     model = AutoEncoder(train_matrix.shape[1], k)
    #
    # # Set optimization hyperparameters.
    #     lr = 0.01
    #     num_epoch = 50
    #
    #     highest_acc = train(model, lr, lamb, train_matrix, zero_train_matrix,
    #         valid_data, num_epoch)
    #     test_result = evaluate(model, zero_train_matrix, test_data)
    #     print("test accuracy: \n" + str(test_result))
    #     lambda_accuracy.append(highest_acc)
    # best_lambda = lambda_accuracy.index(max(lambda_accuracy))
    # print("will use this lambda for further procedure due to its highest accuracy:" + str(best_lambda))

    k = 50
    model = AutoEncoder(train_matrix.shape[1], k)

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 60
    lamb = 0.0001
    # below parameters is for part B experiment
    # lr = 0.001
    # num_epoch = 200
    # lamb = 0.0001

    train(model, lr, lamb, train_matrix, zero_train_matrix,
          train_data, valid_data, num_epoch)
    test_result = evaluate(model, zero_train_matrix, test_data)
    print("test accuracy: \n" + str(test_result))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
