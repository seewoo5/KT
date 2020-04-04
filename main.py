from config import args
import util
from dataset.modified_AAAI20.dataset_modified_AAAI20 import KTDataset
from dataset.dataset_user_sep import UserSepDataset
from network.DKT import DKT
from network.DKVMN import DKVMN
from constant import QUESTION_NUM
from trainer import Trainer
from network.util_network import load_pretrained_weight_DKT
import numpy as np


def run(i):
    """
    i: single integer represents dataset number
    """
    user_base_path = f'{args.base_path}/{args.target_dataset_name}/processed'

    train_data_path = f'{user_base_path}/{i}/train/'
    val_data_path = f'{user_base_path}/{i}/val/'
    test_data_path = f'{user_base_path}/{i}/test/'

    train_sample_infos, num_of_train_user = util.get_data_user_sep(train_data_path)
    val_sample_infos, num_of_val_user = util.get_data_user_sep(val_data_path)
    test_sample_infos, num_of_test_user = util.get_data_user_sep(test_data_path)

    train_data = UserSepDataset('train', train_sample_infos, args.target_dataset_name)
    val_data = UserSepDataset('val', val_sample_infos, args.target_dataset_name)
    test_data = UserSepDataset('test', test_sample_infos, args.target_dataset_name)

    print(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_infos)}')
    print(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_infos)}')
    print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}')

    model = DKT(args.input_dim, args.hidden_dim, args.num_layers, QUESTION_NUM[args.target_dataset_name], args.dropout).to(args.device)
    if args.source_pretrained_weight_path is not None:
        load_pretrained_weight_DKT(model, args.source_freeze, is_source=True)  # from source: LSTM weight
    if args.target_pretrained_weight_path is not None:
        load_pretrained_weight_DKT(model, args.target_freeze, is_source=False)  # from target: encoder & decoder weight

    trainer = Trainer(model, args.device, args.warm_up_step_count,
                  args.hidden_dim, args.num_epochs, args.weight_path,
                  args.lr, train_data, val_data, test_data)
    trainer.train()
    trainer.test(0)
    return trainer.test_acc, trainer.test_auc


if __name__ == '__main__':

    if args.target_dataset_name == 'modified_AAAI20':
        user_base_path = f'{args.base_path}/modified_AAAI20/response/data_tree'
        qid_mapper_path = f'dataset/{args.target_dataset_name}/content_dict.csv'
        qid_to_embed_id = util.get_qid_to_embed_id(qid_mapper_path)
    else:
        user_base_path = f'{args.base_path}/{args.target_dataset_name}/processed'

    if args.target_dataset_name == 'modified_AAAI20':
        train_data_path = f'{args.base_path}/{args.target_dataset_name}/response/new_train_user_list.csv'
        val_data_path = f'{args.base_path}/{args.target_dataset_name}/response/new_val_user_list.csv'
        test_data_path = f'{args.base_path}/{args.target_dataset_name}/response/new_test_user_list.csv'

        train_sample_infos, num_of_train_user = util.get_sample_info(user_base_path, train_data_path)
        val_sample_infos, num_of_val_user = util.get_sample_info(user_base_path, val_data_path)
        test_sample_infos, num_of_test_user = util.get_sample_info(user_base_path, test_data_path)

        train_data = KTDataset('train', user_base_path, train_sample_infos, qid_to_embed_id)
        val_data = KTDataset('val', user_base_path, val_sample_infos, qid_to_embed_id)
        test_data = KTDataset('test', user_base_path, test_sample_infos, qid_to_embed_id)

        print(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_infos)}')
        print(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_infos)}')
        print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}')

        model = DKT(args.input_dim, args.hidden_dim, args.num_layers, QUESTION_NUM[args.target_dataset_name], args.dropout).to(
            args.device)
        if args.source_pretrained_weight_path is not None:
            load_pretrained_weight_DKT(model, args.source_freeze, is_source=True)  # from source: LSTM weight
        if args.target_pretrained_weight_path is not None:
            load_pretrained_weight_DKT(model, args.target_freeze, is_source=False)  # from target: encoder & decoder weight
        trainer = Trainer(model, args.device, args.warm_up_step_count,
                          args.hidden_dim, args.num_epochs, args.weight_path,
                          args.lr, train_data, val_data, test_data)
        trainer.train()
        trainer.test(0)

    else:
        if args.cross_validation is False:
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
