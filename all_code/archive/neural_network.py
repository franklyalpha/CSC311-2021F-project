from all_code.utils import *
from data_pre_process import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

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

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k, l, m):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)
        self.i1 = nn.Linear(k, l)
        self.i2 = nn.Linear(l, m)
        self.i3 = nn.Linear(m, l)
        self.i4 = nn.Linear(l, k)
        self.user_b = torch.rand(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        i1_w_norm = torch.norm(self.i1.weight, 2) ** 2
        i2_w_norm = torch.norm(self.i2.weight, 2) ** 2
        i3_w_norm = torch.norm(self.i3.weight, 2) ** 2
        i4_w_norm = torch.norm(self.i4.weight, 2) ** 2
        return g_w_norm + h_w_norm + i1_w_norm + i2_w_norm + i3_w_norm + i4_w_norm

    def forward(self, inputs, user_bias):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################

        inner = self.g.forward(inputs + user_bias)
        inner_act = nn.Tanh()(inner)
        hidden1 = self.i1.forward(inner_act)
        #hidden1_act = nn.Sigmoid()(hidden1)
        hidden2 = self.i2.forward(hidden1)
        hidden2_act = nn.Sigmoid()(hidden2)
        hidden3 = self.i3.forward(hidden2_act)
        hidden4 = self.i4.forward(hidden3)
        outer = self.h.forward(hidden4)
        outer_activ = nn.Sigmoid()(outer)
        out = outer_activ
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, user_info):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float -> for regularization
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.
    # discuss possibilities: 1: adding the regularization term only once

    # Tell PyTorch you are training the model.
    model.train()
    stu_bias_matrix = pre_process_stu_data(user_info)
    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    valid_acc = 0.
    pass1, pass2 = False, False
    for epoch in range(0, num_epoch):
        train_loss = 0.
        # loss = None
        # if (valid_acc >= 0.70) and not pass1:
        #     new_lr = lr / 10
        #     optimizer = optim.SGD(model.parameters(), lr=new_lr)
        #     pass1 = True
        # if (valid_acc >= 0.697) and not pass2:
        #     new_lr = lr / 5
        #     optimizer = optim.SGD(model.parameters(), lr=new_lr)
        #     pass2 = True
        # norm = model.get_weight_norm()
        # denom = 2
        for user_id in range(num_student):

            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()
            user_bias = torch.from_numpy(stu_bias_matrix[user_id])
            user_bias = user_bias.float()
            optimizer.zero_grad()
            output = model(inputs, user_bias)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # norm = model.get_weight_norm()
            # denom = 2 * num_student
            # loss = torch.sum((output - target) ** 2.) + (lamb / denom) * norm

            loss = torch.sum((output - target) ** 2.)
            # setting regularization at last to reduce final model's extreme values
            # requiring a large lambda
            # this method doesn't work, result in error
            # if user_id == num_student - 1:
            #     loss = torch.sum((output - target) ** 2.) + (lamb / denom) * norm
            # else:
            #     loss = torch.sum((output - target) ** 2.)

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data, stu_bias_matrix)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data, stu_bias_matrix):
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
        user_bias = torch.from_numpy(stu_bias_matrix[u])
        user_bias = user_bias.float()
        output = model(inputs, user_bias)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return round(correct / float(total), 4)


def main():

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.

    k = 200
    l = 50
    m = 10
    model = AutoEncoder(train_matrix.shape[1], k, l, m)
    user_data = fill_null_data_user()
    # Set optimization hyperparameters.
    lr = 0.001
    num_epoch = 500
    lamb = 0.0001
    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch, user_data)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()


