from config import args
import util
from dataset import KTDataset
from torch.utils.data import DataLoader
from network.DKT import DKT
from constant import *
import tqdm
import torch.nn as nn
from trainer import Trainer


if __name__ == '__main__':

    qid_mapper_path = 'load/content_dict.csv'
    qid_to_embed_id = util.get_qid_to_embed_id(qid_mapper_path)
    user_base_path = f'{args.base_path}/modified_AAAI20/response/data_tree'

    if args.debug_mode:
        train_data_path = 'load/sample_train.csv'
        val_data_path = 'load/sample_val.csv'
        test_data_path = 'load/sample_test.csv'
    else:
        train_data_path = f'{args.base_path}/modified_AAAI20/response/new_train_user_list.csv'
        val_data_path = f'{args.base_path}/modified_AAAI20/response/new_val_user_list.csv'
        test_data_path = f'{args.base_path}/modified_AAAI20/response/new_test_user_list.csv'

    train_sample_infos, num_of_train_user = util.get_sample_info(user_base_path, train_data_path)
    val_sample_infos, num_of_val_user = util.get_sample_info(user_base_path, val_data_path)
    test_sample_infos, num_of_test_user = util.get_sample_info(user_base_path, test_data_path)

    print(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_infos)}')
    print(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_infos)}')
    print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}')

    train_data = KTDataset('train', user_base_path, train_sample_infos, qid_to_embed_id, True)
    val_data = KTDataset('val', user_base_path, val_sample_infos, qid_to_embed_id, False)
    test_data = KTDataset('test', user_base_path, test_sample_infos, qid_to_embed_id, False)

    # TODO: input dimension may differ from hidden dimension (d_model), change to one-hot
    model = DKT(args.d_model, args.d_model, args.num_layers, QUESTION_NUM, args.dropout).to(args.device)

    trainer = Trainer(model, args.device, args.warm_up_step_count,
                      args.d_model, args.num_epochs, args.weight_path,
                      args.lr, train_data, val_data, test_data)
    trainer.train()
    trainer.test(0)
