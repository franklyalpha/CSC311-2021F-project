import numpy as np
from utils import *
import pdb

# the idea of classifying data for similar users:
'''
take the mean of values for birth date and 0.5 for premium pupil, then use that for calculation. 
will classify training data by gender first. If a gender is unspecified, 
it will be set to 1.5 to avoid data bias.
vectorize each user's point 
then find 10 closest user on 3D space, take their data into account. 
predict by threshold: will apply a bias of each question to a ratio of correctness or 1. 

'''


def pre_process_stu_data(student):
    """
    return a bias matrix where each row represents the student's bias regarding each question's answer

    """
    train_matrix = np.array(load_train_sparse("../data").toarray())
    train_shape = train_matrix.shape
    stu_bias_matrix = np.empty((train_shape[0], train_shape[1]))
    user_id_np = np.array(student["user_id"])
    for user_id in student["user_id"]:
        closest_user_index = find_similar_users(student, user_id)
        real_user = user_id_np[closest_user_index]
        segment_train = np.take(train_matrix, real_user, axis=0)
        correctness = np.nanmean(segment_train, axis=0)
    # if correctness is > 0.5, this means more students got this one correct.
    # thus subtract the value by 0.5, and then multiply by 2 gives ratio of answering
        student_bias = (correctness - 0.5) * 2
        student_bias = np.nan_to_num(student_bias)
        stu_bias_matrix[user_id, :] = student_bias
    return stu_bias_matrix


def fill_null_data_user():
    """
    return a dictionary of students containing:
    an "user_id" array
    a "gender" array
    a "date_of_birth" array
    a "premium_pupil" array
    where each array's element corresponds to the user's information at same position
    of "user_id" array.


    """

    student = read_stu_meta()

    date = normalize_date(student["date_of_birth"])
    student["date_of_birth"] = date
    # set unspecified gender as 1.5
    gender_list = student["gender"]
    for index in range(len(gender_list)):
        if gender_list[index] == 0:
            gender_list[index] = 1.5
    premium = student["premium_pupil"]
    for index in range(len(premium)):
        if premium[index] is None:
            premium[index] = 0.5
    dates = student["date_of_birth"]
    date_mean = np.nanmean(np.array(dates))
    # since the similarity of students won't be considered by
    # neural network (they just present as bias), can replace values
    for index in range(len(dates)):
        if np.isnan(dates[index]):
            dates[index] = date_mean
    return student


def find_similar_users(user_dict, input_user_id):
    """
    return 20 user id having the most similar background with "input_user_id"'s information.
    """
    # formula for determining relative distance:
    # value of input subtract all data (regardless of whether itself is chosen) and then squared
    users_id = user_dict["user_id"]
    users_gender = np.array(user_dict["gender"])
    users_date_birth = np.array(user_dict["date_of_birth"])
    users_premium = np.array(user_dict["premium_pupil"])
    user_index = users_id.index(input_user_id)
    user_info = [users_gender[user_index],
                 users_date_birth[user_index], users_premium[user_index]]
    final_distance = np.subtract(user_info[0], users_gender) ** 2 + \
                     np.subtract(user_info[1], users_date_birth) ** 2 + \
                     np.subtract(user_info[2], users_premium) ** 2
    final_distance = np.sqrt(final_distance)
    # now find out 10 closest users from here
    top_k_index = np.argsort(final_distance)
    return top_k_index[:20]


def create_stu_meta_matrix():
    """Create a 2D numpy array of student meta data that includes already 
    normalized student information, with the rows of the matrix sorted based on
    student ID in increasing order. 
    :return: 2D numpy array of student meta data sorted in increasing order of
            student ID.
    """
    # read in normalized student data
    stu_dict = fill_null_data_user()
    # create a 2D np array of student data
    stu_matrix = np.column_stack((stu_dict["user_id"], stu_dict["gender"], \
        stu_dict["date_of_birth"], stu_dict["premium_pupil"]))
    # sort the np array by student ID in increasing order
    stu_matrix_sorted = stu_matrix[np.argsort(stu_matrix[:, 0])]
    return stu_matrix_sorted
