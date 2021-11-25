import numpy as np
from utils import *

# the idea of classifying data for similar users:
'''
take the mean of values for birth date and 0.5 for premium pupil, then use that for calculation. 
will classify training data by gender first. If a gender is unspecified, 
it will be set to 1.5 to avoid data bias.
vectorize each user's point 
then find 10 closest user on 3D space, take their data into account. 
predict by threshold: will apply a bias of each question to a ratio of correctness or 1. 

'''


def pre_process_stu_data():
    student = fill_null_data_user()
    closest_user_index = find_similar_users(student, 32)
    train_matrix = load_train_sparse("../data").toarray()
    segment_train = np.take(train_matrix, [closest_user_index], axis=0)
    correctness = np.nanmean(segment_train, axis=1)
    # if correctness is > 0.5, this means more students got this one correct.
    # thus subtract the value by 0.5, and then multiply by 2 gives ratio of answering


def fill_null_data_user():
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
