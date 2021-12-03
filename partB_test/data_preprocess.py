import numpy as np
from utils import *
import pdb

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
    
if __name__ == "__main__":
    create_student_matrix()