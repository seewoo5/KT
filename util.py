import csv
import glob
import os


def create_full_path(user_base_path, user_path):
    u0 = user_path[0]
    u1 = user_path[1]
    u2 = user_path[2]
    u3 = user_path[3]
    return f'{user_base_path}/{u0}/{u1}/{u2}/{u3}/{user_path}'


def get_qid_to_embed_id(dict_path):
    d = {}
    with open(dict_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split(',')
            d[int(line[0])] = int(line[1])
    return d


def get_sample_info(user_base_path, data_path):
    # for modified_AAAI20 data
    sample_infos = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        num_of_users = len(lines)
        for user_path in lines:
            user_path = user_path.rstrip()
            user_full_path = create_full_path(user_base_path, user_path)
            with open(user_full_path, 'r', encoding='ISO-8859-1') as f:
                lines = f.readlines()
                num_of_interactions = len(lines)
                for target_index in range(num_of_interactions):
                    sample_infos.append([user_path, target_index])

    return sample_infos, num_of_users

# Do not use this anymore
def get_data_tl(data_path):
    # for triple line format data
    data_list = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        num_of_users = len(lines) // 3
        for i in range(num_of_users):
            user_interaction_len = int(lines[3*i].strip())
            qid_list = list(map(int, lines[3*i+1].split(',')))
            is_correct_list = list(map(int, lines[3*i+2].split(',')))
            assert user_interaction_len == len(qid_list) == len(is_correct_list), 'length is different'

            for j in range(user_interaction_len):
                data_list.append((qid_list[:j+1], is_correct_list[:j+1]))

    return data_list, num_of_users


def get_data_user_sep(data_path):
    # almost same as get_sample_info
    # for user separated format data
    sample_infos = []
    # get list of all files
    user_path_list = os.listdir(data_path)
    num_of_users = len(user_path_list)

    for user_path in user_path_list:
        with open(data_path + user_path, 'rb') as f:
            lines = f.readlines()
            lines = lines[1:]
            num_of_interactions = len(lines)
            for end_index in range(num_of_interactions):
                sample_infos.append((data_path + user_path, end_index))

    return sample_infos, num_of_users
