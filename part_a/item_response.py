from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.

    for i, uid in enumerate(data["user_id"]):
        qid = data["question_id"][i]
        c = data["is_correct"][i]
        log_lklihood += c * (theta[uid] - beta[qid]) - np.log(
            1 + np.exp(theta[uid] - beta[qid]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    uids = np.array(data["user_id"])
    qids = np.array(data["question_id"])
    cs = np.array(data["is_correct"])

    theta_copy = theta.copy()
    beta_copy = beta.copy()

    # update theta
    # here notice that the equation is different from what we have in part a
    # in part a: derivative with respect to theta_i is Sum(cij - sigmoid(..))
    # but here we are doing Sum(sigmoid(..) - cij)
    # this is because we are using negative lld here while in part a we are
    # calculating the derivative for lld
    for i in range(len(theta)):
        # we only consider the questions that user i answers
        sigmoid_part = sigmoid(theta_copy[i] - beta_copy[qids[uids == i]])
        theta[i] -= lr * (np.sum(sigmoid_part) - np.sum(cs[uids == i]))
    # update beta
    for j in range(len(beta)):
        # we only consider the users who answer question j
        sigmoid_part = sigmoid(theta_copy[uids[qids == j]] - beta_copy[j])
        beta[j] -= lr * (np.sum(cs[qids == j]) - np.sum(sigmoid_part))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(train_data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros([542, 1])
    beta = np.zeros([1774, 1])

    train_acc_lst = []
    val_acc_lst = []
    train_neg_log_likelihood = []
    val_neg_log_likelihood = []

    for i in range(iterations):
        theta, beta = update_theta_beta(train_data, lr, theta, beta)

        train_neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta)
        train_neg_log_likelihood.append(train_neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_neg_log_likelihood.append(val_neg_lld)

        train_score = evaluate(data=train_data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)

        print("TRAIN - NLLK: {} \t Score: {}".format(train_neg_lld,
                                                     train_score))
        print("VAL - NLLK: {} \t Score: {}".format(val_neg_lld, val_score))

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_neg_log_likelihood, val_neg_log_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 7

    theta, beta, train_neg_log_likelihood, val_neg_log_likelihood = irt(
        train_data, val_data, lr, iterations)

    plt.plot(train_neg_log_likelihood, label="training")
    plt.plot(val_neg_log_likelihood, label="validation")
    plt.ylabel("Negative Log-Likelihood")
    plt.xlabel("Number of Iterations")
    plt.title("Negative Log-Likelihood for Training and Validation Data")
    plt.legend()
    plt.savefig("Q2b.png")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # part c
    val_score = evaluate(data=val_data, theta=theta, beta=beta)
    test_score = evaluate(data=test_data, theta=theta, beta=beta)

    print("Validation Accuracy: ", val_score)
    print("Test Accuracy: ", test_score)

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    # we have 1774 questions in total. Select three:
    questions = [64, 734, 1356]
    # we sort the theta for the sake a=of plotting
    theta_copy = theta.reshape(-1)
    theta_copy.sort()

    for qid in questions:
        prob = sigmoid(theta_copy - beta[qid])
        plt.plot(theta_copy, prob, label="Question {} with beta {:.2f}".format(
            qid, beta[qid][0]))

    plt.ylabel("Probability of the Correct Response")
    plt.xlabel("Theta")
    plt.title(
        "Probability of the Correct Response to Selected Questions vs. Theta")
    plt.legend()
    plt.savefig("Q2d.png")
    plt.show()

#####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
