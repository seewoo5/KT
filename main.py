from config import ARGS
import util
# from dataset.modified_AAAI20.dataset_modified_AAAI20 import KTDataset
from dataset.dataset_user_sep import UserSepDataset
from network.DKT import DKT
from network.DKVMN import DKVMN
from network.NPA import NPA
from constant import QUESTION_NUM
from trainer import Trainer
from network.util_network import load_pretrained_weight_DKT
import numpy as np


def get_model():
    if ARGS.model == 'DKT':
        model = DKT(ARGS.input_dim, ARGS.hidden_dim, ARGS.num_layers, QUESTION_NUM[ARGS.target_dataset_name],
                    ARGS.dropout).to(ARGS.device)
        if ARGS.source_pretrained_weight_path is not None:
            load_pretrained_weight_DKT(model, ARGS.source_freeze, is_source=True)  # from source: LSTM weight
        if ARGS.target_pretrained_weight_path is not None:
            load_pretrained_weight_DKT(model, ARGS.target_freeze, is_source=False)  # from target: encoder & decoder weight

    elif ARGS.model == 'DKVMN':
        model = DKVMN(ARGS.key_dim, ARGS.value_dim, ARGS.summary_dim, QUESTION_NUM[ARGS.target_dataset_name],
                      ARGS.concept_num).to(ARGS.device)

    elif ARGS.model == 'NPA':
        model = NPA(ARGS.input_dim, ARGS.hidden_dim, ARGS.attention_dim, ARGS.fc_dim,
                    ARGS.num_layers, QUESTION_NUM[ARGS.target_dataset_name], ARGS.dropout).to(ARGS.device)

    else:
        raise NotImplementedError

    return model


def run(i):
    """
    i: single integer represents dataset number
    """
    user_base_path = f'{ARGS.base_path}/{ARGS.target_dataset_name}/processed'

    train_data_path = f'{user_base_path}/{i}/train/'
    val_data_path = f'{user_base_path}/{i}/val/'
    test_data_path = f'{user_base_path}/{i}/test/'

    train_sample_infos, num_of_train_user = util.get_data_user_sep(train_data_path)
    val_sample_infos, num_of_val_user = util.get_data_user_sep(val_data_path)
    test_sample_infos, num_of_test_user = util.get_data_user_sep(test_data_path)

    train_data = UserSepDataset('train', train_sample_infos, ARGS.target_dataset_name)
    val_data = UserSepDataset('val', val_sample_infos, ARGS.target_dataset_name)
    test_data = UserSepDataset('test', test_sample_infos, ARGS.target_dataset_name)

    print(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_infos)}')
    print(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_infos)}')
    print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}')

    model = get_model()

    trainer = Trainer(model, ARGS.device, ARGS.warm_up_step_count,
                      ARGS.hidden_dim, ARGS.num_epochs, ARGS.weight_path,
                      ARGS.lr, train_data, val_data, test_data)
    trainer.train()
    trainer.test(0)
    return trainer.test_acc, trainer.test_auc


if __name__ == '__main__':

    # if ARGS.target_dataset_name == 'modified_AAAI20':
    #     user_base_path = f'{ARGS.base_path}/modified_AAAI20/response/data_tree'
    #     qid_mapper_path = f'dataset/{ARGS.target_dataset_name}/content_dict.csv'
    #     qid_to_embed_id = util.get_qid_to_embed_id(qid_mapper_path)
    # else:
    #     user_base_path = f'{ARGS.base_path}/{ARGS.target_dataset_name}/processed'

    # if ARGS.target_dataset_name == 'modified_AAAI20':
    #     train_data_path = f'{ARGS.base_path}/{ARGS.target_dataset_name}/response/new_train_user_list.csv'
    #     val_data_path = f'{ARGS.base_path}/{ARGS.target_dataset_name}/response/new_val_user_list.csv'
    #     test_data_path = f'{ARGS.base_path}/{ARGS.target_dataset_name}/response/new_test_user_list.csv'
    #
    #     train_sample_infos, num_of_train_user = util.get_sample_info(user_base_path, train_data_path)
    #     val_sample_infos, num_of_val_user = util.get_sample_info(user_base_path, val_data_path)
    #     test_sample_infos, num_of_test_user = util.get_sample_info(user_base_path, test_data_path)
    #
    #     train_data = KTDataset('train', user_base_path, train_sample_infos, qid_to_embed_id)
    #     val_data = KTDataset('val', user_base_path, val_sample_infos, qid_to_embed_id)
    #     test_data = KTDataset('test', user_base_path, test_sample_infos, qid_to_embed_id)
    #
    #     print(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_infos)}')
    #     print(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_infos)}')
    #     print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}')
    #
    #     model = DKT(ARGS.input_dim, ARGS.hidden_dim, ARGS.num_layers, QUESTION_NUM[ARGS.target_dataset_name], ARGS.dropout).to(
    #         ARGS.device)
    #     if ARGS.source_pretrained_weight_path is not None:
    #         load_pretrained_weight_DKT(model, ARGS.source_freeze, is_source=True)  # from source: LSTM weight
    #     if ARGS.target_pretrained_weight_path is not None:
    #         load_pretrained_weight_DKT(model, ARGS.target_freeze, is_source=False)  # from target: encoder & decoder weight
    #     trainer = Trainer(model, ARGS.device, ARGS.warm_up_step_count,
    #                       ARGS.hidden_dim, ARGS.num_epochs, ARGS.weight_path,
    #                       ARGS.lr, train_data, val_data, test_data)
    #     trainer.train()
    #     trainer.test(0)

    if ARGS.cross_validation is False:
        test_acc, test_auc = run(1)
    else:
        acc_list = []
        auc_list = []

        for i in range(1, 6):
            print(f'{i}th dataset')
            test_acc, test_auc = run(i)
            acc_list.append(test_acc)
            auc_list.append(test_auc)

        acc_array = np.asarray(acc_list)
        auc_array = np.asarray(auc_list)
        print(f'mean acc: {np.mean(acc_array):.4f}, auc: {np.mean(auc_array):.4f}')
        print(f'std acc: {np.std(acc_array):.4f}, auc: {np.std(auc_array):.4f}')
    # else:
    # if ARGS.cross_validation is False:
    #     test_acc, test_auc = run(1)
    # else:
    #         acc_list = []
    #         auc_list = []
    #         for i in range(1, 6):
    #             print(f'{i}th dataset')
    #             test_acc, test_auc = run(i)
    #             acc_list.append(test_acc)
    #             auc_list.append(test_auc)
    #
    #         acc_array = np.asarray(acc_list)
    #         auc_array = np.asarray(auc_list)
    #         print(f'mean acc: {np.mean(acc_array):.4f}, auc: {np.mean(auc_array):.4f}')
    #         print(f'std acc: {np.std(acc_array):.4f}, auc: {np.std(auc_array):.4f}')
