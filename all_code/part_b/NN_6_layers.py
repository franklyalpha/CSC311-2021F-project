from all_code.utils import *
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
    def __init__(self, num_question, code1, code2, code3, code_vect):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, code1)
        self.encode1 = nn.Linear(code1, code2)
        self.encode2 = nn.Linear(code2, code3)
        self.encode3 = nn.Linear(code3, code_vect)
        self.decode1 = nn.Linear(code_vect, code3)
        self.decode2 = nn.Linear(code3, code2)
        self.decode3 = nn.Linear(code2, code1)
        self.h = nn.Linear(code1, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        en1_w_norm = torch.norm(self.encode1.weight, 2) ** 2
        en2_w_norm = torch.norm(self.encode2.weight, 2) ** 2
        en3_n = torch.norm(self.encode3.weight, 2) ** 2
        de1_w_norm = torch.norm(self.decode1.weight, 2) ** 2
        de2_w_norm = torch.norm(self.decode2.weight, 2) ** 2
        de3_n = torch.norm(self.decode3.weight, 2) ** 2
        return g_w_norm + h_w_norm + en1_w_norm + en2_w_norm + de1_w_norm + de2_w_norm + en3_n + de3_n

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
        inner_act = nn.Tanh()(inner)
        hidden1 = self.encode1.forward(inner_act)
        hidden2 = self.encode2.forward(hidden1)
        hidden3 = self.encode3.forward(hidden2)
        hidden3_act = nn.Sigmoid()(hidden3)
        hidden4 = self.decode1.forward(hidden3_act)
        hidden5 = self.decode2.forward(hidden4)
        hidden6 = self.decode3.forward(hidden5)
        outer = self.h.forward(hidden6)
        outer_activ = nn.Sigmoid()(outer)
        out = outer_activ
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
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

    # Tell PyTorch you are training the model.
    model.train()
    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    # valid_acc_record = []

    for epoch in range(0, num_epoch):
        train_loss = 0.
        for user_id in range(num_student):

            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()
            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            norm = model.get_weight_norm()
            denom = 2
            loss = torch.sum((output - target) ** 2.) + (lamb / denom) * norm
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    #     valid_acc_record.append(valid_acc)
    #
    # with open("nn6_valid_acc.txt", "w") as fp:
    #     json.dump(valid_acc_record, fp)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


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
    return round(correct / float(total), 4)


def main():

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.

    code1 = 150
    code2 = 80
    code3 = 30
    code_vect = 10
    model = AutoEncoder(train_matrix.shape[1], code1, code2, code3, code_vect)

    # Set optimization hyperparameters.
    lr = 0.001
    num_epoch = 200
    lamb = 0.0001
    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    test_result = evaluate(model, zero_train_matrix, test_data)
    print("test accuracy: \n" + str(test_result))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()


