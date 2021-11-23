from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.transpose())
    acc = sparse_matrix_evaluate(valid_data, mat.transpose())
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def _plot(k_vals, accs, filename):
    plt.clf()
    plt.plot(k_vals, accs, marker="o")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.savefig(filename)


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_vals = [1, 6, 11, 16, 21, 26]
    # 1(a)
    user_accs = []
    for k in k_vals:
        user_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_accs.append(user_acc)
    
    # Plot and report the accuracy on the validation data as a function of k
    _plot(k_vals, user_accs, "accuracy_user_based.png")
    
    # 1(b)
    # k=11 has the highest performance on validation data
    user_acc_test = knn_impute_by_user(sparse_matrix, test_data, 11)
    print(f"The test accuracy for user-based collaborative filtering kNN is: \
        {user_acc_test}")

    # 1(c)
    item_accs = []
    for k in k_vals:
        item_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_accs.append(item_acc)
    
    # Plot and report the accuracy on the validation data as a function of k
    _plot(k_vals, item_accs, "accuracy_item_based.png")
    
    # k=21 has the highest performance on validation data
    item_acc_test = knn_impute_by_item(sparse_matrix, test_data, 21)
    print(f"The test accuracy for item-based collaborative filtering kNN is: \
        {item_acc_test}")
        
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
