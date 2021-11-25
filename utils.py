import datetime

from scipy.sparse import load_npz

import numpy as np
import csv
import os
import datetime

import typing

from typing import List
from typing import Optional


def _load_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["user_id"].append(int(row[1]))
                data["is_correct"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_train_sparse(root_dir="/data"):
    """ Load the training data as a spare matrix representation.

    :param root_dir: str
    :return: 2D sparse matrix
    """
    path = os.path.join(root_dir, "train_sparse.npz")
    if not os.path.exists(path):
        raise Exception("The specified path {} "
                        "does not exist.".format(os.path.abspath(path)))
    matrix = load_npz(path)
    return matrix


def load_train_csv(root_dir="/data"):
    """ Load the training data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "train_data.csv")
    return _load_csv(path)


def load_valid_csv(root_dir="/data"):
    """ Load the validation data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "valid_data.csv")
    return _load_csv(path)


def load_public_test_csv(root_dir="/data"):
    """ Load the test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "test_data.csv")
    return _load_csv(path)


def load_private_test_csv(root_dir="/data"):
    """ Load the private test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: an empty list.
    """
    path = os.path.join(root_dir, "private_test_data.csv")
    return _load_csv(path)


def save_private_test_csv(data, file_name="private_test_result.csv"):
    """ Save the private test data as a csv file.

    This should be your submission file to Kaggle.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    :param file_name: str
    :return: None
    """
    if not isinstance(data, dict):
        raise Exception("Data must be a dictionary.")
    cur_id = 1
    valid_id = ["0", "1"]
    with open(file_name, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "is_correct"])
        for i in range(len(data["user_id"])):
            if str(int(data["is_correct"][i])) not in valid_id:
                raise Exception("Your data['is_correct'] is not in a valid format.")
            writer.writerow([str(cur_id), str(int(data["is_correct"][i]))])
            cur_id += 1
    return


def evaluate(data, predictions, threshold=0.5):
    """ Return the accuracy of the predictions given the data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param predictions: list
    :param threshold: float
    :return: float
    """
    if len(data["is_correct"]) != len(predictions):
        raise Exception("Mismatch of dimensions between data and prediction.")
    if isinstance(predictions, list):
        predictions = np.array(predictions).astype(np.float64)
    return (np.sum((predictions >= threshold) == data["is_correct"])
            / float(len(data["is_correct"])))


def sparse_matrix_evaluate(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def sparse_matrix_predictions(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the predictions.

    This function can be used for submitting Kaggle competition.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: list
    """
    predictions = []
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold:
            predictions.append(1.)
        else:
            predictions.append(0.)
    return predictions


def read_meta(root_dir="data/"):
    """
    Returns reading result of student_meta.csv and question_meta.csv as a tuple.
    """
    stu_meta_path = os.path.join(root_dir, "student_meta.csv")
    ques_meta_path = os.path.join(root_dir, "question_meta.csv")
    stu_data = {
        "user_id": [],
        "gender": [],
        "date_of_birth": [],
        "premium_pupil": []
    }
    que_data = {
        "question_id": [],
        "subject_id": []
    }
    return (_read_stu_meta(stu_data, stu_meta_path),
            _read_que_meta(que_data, ques_meta_path))


def _read_stu_meta(stu_data, stu_meta_path):
    with open(stu_meta_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        line_count = 0
        for row in reader:
            if line_count == 0:
                line_count += 1
                continue
            stu_data["user_id"].append(int(row[0]))
            stu_data["gender"].append(int(row[1]))
            if row[2] is not "":
                date = datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S.%f")
                stu_data["date_of_birth"].append(date)
            else:
                stu_data["date_of_birth"].append(None)
            if row[3] is not "":
                stu_data["premium_pupil"].append(float(row[3]))
            else:
                stu_data["premium_pupil"].append(None)
    return stu_data


def _read_que_meta(que_data, que_meta_path):
    with open(que_meta_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        line_count = 0
        for row in reader:
            if line_count == 0:
                line_count += 1
                continue
            que_data["question_id"].append(int(row[0]))
            subjects = []
            tmp_num = ""
            for chars in row[1]:
                if chars in "[] ,":
                    if tmp_num is not "":
                        subjects.append(int(tmp_num))
                    tmp_num = ""
                    continue
                tmp_num += chars
            que_data["subject_id"].append(subjects)
    return que_data


def revert_subj_ques_order(subject_list: List[List[int]]):
    """
    return a dictionary where key is subject_id and value is
    a list of question_id belonging to subject with subject_id.
    """
    sub_que_dict = {}
    for ques_index in range(len(subject_list)):
        ques_related_sub = subject_list[ques_index]
        for subject_id in ques_related_sub:
            if subject_id not in sub_que_dict:
                sub_que_dict[subject_id] = [ques_index]
            else:
                sub_que_dict[subject_id].append(ques_index)
    return sub_que_dict


def normalize_date(date_of_birth: List[Optional[datetime.datetime]]):
    min_d = min(x for x in date_of_birth if x is not None)
    max_d = max(x for x in date_of_birth if x is not None)
    largest_diff = max_d - min_d
    normalized_date = []
    for dates in date_of_birth:
        if dates is None:
            normalized_date.append(np.NAN)
            continue
        else:
            normalization = (dates - min_d) / largest_diff
            normalized_date.append(normalization)
    return normalized_date