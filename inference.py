from config import ARGS
import util
from dataset.modified_AAAI20.dataset_modified_AAAI20 import KTDataset
from dataset.dataset_tl import TripleLineDataset
from network.DKT import DKT
from constant import *
from trainer import Trainer


if __name__ == '__main__':

    qid_mapper_path = 'load/content_dict.csv'
    qid_to_embed_id = util.get_qid_to_embed_id(qid_mapper_path)
    user_base_path = f'{ARGS.base_path}/modified_AAAI20/response/data_tree'

    if ARGS.debug_mode:
        test_data_path = 'load/sample_test_modified_AAAI20.csv'
    else:
        if ARGS.dataset_name == 'modified_AAAI20':
            test_data_path = f'{ARGS.base_path}/modified_AAAI20/response/new_test_user_list.csv'
            test_sample_infos, num_of_test_user = util.get_sample_info(user_base_path, test_data_path)
            test_data = KTDataset('test', user_base_path, test_sample_infos, qid_to_embed_id, False)
        else:
            test_data_path = f'dataset/{ARGS.dataset_name}/test_data.csv'
            test_sample_infos, num_of_test_user = util.get_data(test_data_path)
            test_data = TripleLineDataset('test', test_sample_infos)

    print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}')

    model = DKT(ARGS.d_model, ARGS.d_model, ARGS.num_layers, QUESTION_NUM, ARGS.dropout).to(ARGS.device)
    weight_path = f'{ARGS.weight_path}{ARGS.weight_num}.pt'
    trainer = Trainer(model, ARGS.device, ARGS.warm_up_step_count,
                      ARGS.d_model, ARGS.num_epochs, ARGS.weight_path,
                      ARGS.lr, None, None, test_data)
    trainer.test(ARGS.weight_num)
