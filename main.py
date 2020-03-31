from config import args
import util
from dataset.modified_AAAI20.dataset_modified_AAAI20 import KTDataset
from dataset.ASSISTments2009.dataset_assistments import ASSISTDataset
from network.DKT import DKT
from constant import *
from trainer import Trainer



if __name__ == '__main__':

    # qid_mapper_path = 'load/content_dict.csv'
    qid_mapper_path = f'dataset/{args.dataset_name}/content_dict.csv'
    qid_to_embed_id = util.get_qid_to_embed_id(qid_mapper_path)
    if args.dataset_name == 'modified_AAAI20':
        user_base_path = f'{args.base_path}/modified_AAAI20/response/data_tree'
    else:
        user_base_path = f'{args.base_path}/{args.dataset_name}/response/'

    if args.debug_mode:
        train_data_path = f'dataset/{args.dataset_name}/sample_train_{args.dataset_name}.csv'
        val_data_path = f'dataset/{args.dataset_name}/sample_val_{args.dataset_name}.csv'
        test_data_path = f'dataset/{args.daataset_name}/sample_test_{args.dataset_name}.csv'

    else:
        if args.dataset_name == 'modified_AAAI20':
            train_data_path = f'{args.base_path}/{args.dataset_name}/response/new_train_user_list.csv'
            val_data_path = f'{args.base_path}/{args.dataset_name}/response/new_val_user_list.csv'
            test_data_path = f'{args.base_path}/{args.dataset_name}/response/new_test_user_list.csv'

            train_sample_infos, num_of_train_user = util.get_sample_info(user_base_path, train_data_path)
            val_sample_infos, num_of_val_user = util.get_sample_info(user_base_path, val_data_path)
            test_sample_infos, num_of_test_user = util.get_sample_info(user_base_path, test_data_path)

            train_data = KTDataset('train', user_base_path, train_sample_infos, qid_to_embed_id, True)
            val_data = KTDataset('val', user_base_path, val_sample_infos, qid_to_embed_id, False)
            test_data = KTDataset('test', user_base_path, test_sample_infos, qid_to_embed_id, False)
        else:
            train_data_path = f'dataset/{args.dataset_name}/train_data.csv'
            val_data_path = f'dataset/{args.dataset_name}/val_data.csv'
            test_data_path = f'dataset/{args.dataset_name}/test_data.csv'

            train_sample_infos, num_of_train_user = util.get_data(train_data_path)
            val_sample_infos, num_of_val_user = util.get_data(val_data_path)
            test_sample_infos, num_of_test_user = util.get_data(test_data_path)

            train_data = ASSISTDataset('train', train_sample_infos)
            val_data = ASSISTDataset('val', val_sample_infos)
            test_data = ASSISTDataset('test', test_sample_infos)

    print(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_infos)}')
    print(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_infos)}')
    print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}')



    # TODO: input dimension may differ from hidden dimension (d_model), change to one-hot
    qnum = question_num[args.dataset_name]
    model = DKT(args.input_dim, args.hidden_dim, args.num_layers, qnum, args.dropout).to(args.device)

    trainer = Trainer(model, args.device, args.warm_up_step_count,
                      args.hidden_dim, args.num_epochs, args.weight_path,
                      args.lr, train_data, val_data, test_data)
    trainer.train()
    trainer.test(0)
