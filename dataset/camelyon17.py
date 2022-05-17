from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import argparse
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset
import numpy as np
import torch 

class CustomInputWILDSSubset(WILDSSubset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        if isinstance(idx, slice) :
            #Get the start, stop, and step from the slice
            x, y = torch.stack([self[ii] for ii in range(*idx.indices(len(self)))], dim=0)
        else:
            x, y, _ = self.dataset[idx]
        return x, y

    def __len__(self):
        return len(self.dataset)

class Camelyon17(object):
    def __init__(self, config, test_finetune_size = 5000, test_unlabled_size=100):
        super(Camelyon17, self).__init__()
        self.config = config
        self.num_train_evns = 3
        self.input_dim = 96 * 96 * 3
        self.num_class = 2
        dataset = get_dataset(dataset="camelyon17", root_dir = config.data_dir, download=True)

        torch_transform = transforms.Compose(
                [transforms.Resize((96, 96)), transforms.ToTensor()]
            )
        
        train_data = dataset.get_subset(
            "train",
            transform = torch_transform,
        )

        val_data = dataset.get_subset(
            "val",
            transform = torch_transform,
        )

        test_data = dataset.get_subset(
            "test",
            transform = torch_transform,
        )

        # train
        self.train_data_list = []
        train_data_indices = train_data.indices
        all_possible_indices = set((train_data.metadata_array.numpy())[:,0])
        train_meta_array = train_data.metadata_array.numpy()
        for id in all_possible_indices:
            id_mask = train_meta_array[:,0] == id
            env_idx = np.where(id_mask)[0]
            train_env_idx = train_data_indices[env_idx]
            np.random.shuffle(train_env_idx)
            temp_train_dataset = WILDSSubset(dataset, train_env_idx, torch_transform)
            self.train_data_list.append(
                CustomInputWILDSSubset(temp_train_dataset),
            )

        # val
        self.val_data_list = [
            CustomInputWILDSSubset(val_data),
        ]

        # test
        n_tests = len(test_data)
        print(n_tests)
        fine_tune_frac = test_finetune_size / n_tests

        test_data_finetune = dataset.get_subset(
            "test",
            transform = torch_transform,
            frac = fine_tune_frac
        )
        print("actual fine tune size: " + str(len(test_data_finetune)))

        self.test_data_finetune = CustomInputWILDSSubset(test_data_finetune)
        
        unlabled_tune_frac = test_unlabled_size / n_tests

        test_data_unlabled = dataset.get_subset(
            "test",
            transform = torch_transform,
            frac = unlabled_tune_frac
        )
        print("actual unlabled size: " + str(len(test_data_unlabled)))

        self.test_data_unlabled = test_data_unlabled

        self.test_data_list = CustomInputWILDSSubset(test_data)

    def sample_envs(self, env_ind=0, train_val_test = 0):
        # train
        if train_val_test == 0:
            return self.train_data_list[env_ind]

        # val
        if train_val_test == 1:
            return self.val_data_list[env_ind]

        if train_val_test == 2:
            return self.test_data_finetune, self.test_data_unlabled, self.test_data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_envs', type=int, default= 5, help='number of enviroments per training epoch')
    parser.add_argument('--batch_size', type=int, default= 128, help='batch size')
    parser.add_argument('--irm_reg_lambda', type=float, default= 573.6152510448682, help='regularization coeff for irm')
    parser.add_argument('--reg_lambda', type=float, default= 1000, help='regularization coeff for adaptive invariant learning')
    parser.add_argument('--reg_lambda_2', type=float, default= 0.4, help='second regularization coeff for adaptive invariant learning')
    parser.add_argument('--gamma', type=float, default= 0.9, help='interpolation parmameter')
    parser.add_argument('--phi_odim',  type=int, default= 3, help='Phi output size')

    # different models
    parser.add_argument('--model_name', type=str, default= "adp_invar", help='type of modesl. current support: adp_invar, erm')
    parser.add_argument('--compare_all_invariant_models', action='store_true', help='compare all invariant models')
    parser.add_argument('--classification', action='store_true', help='if the tast is classification, set this flag to enable correct prediction, labels has to be between [0, ..., n]')

    # finetune
    parser.add_argument('--run_fine_tune_test', action='store_true', help='run finetunning tests')
    parser.add_argument('--n_fine_tune_tests', type=int, default= 10, help='number of fine tunning tests')
    parser.add_argument('--n_fine_tune_points', nargs='+', type=int, help='how many points for finetuning')

    # dataset
    parser.add_argument('--dataset', type=str, default= "syn", help='type of experiment: syn, bike, color_mnist')
    
    # synthetic dataset specifics
    parser.add_argument('--causal_dir_syn', type=str, default= "anti", help='anti or causal')
    parser.add_argument('--syn_dataset_train_size', type=int, default= 1024, help='size of synthetic dataset per env')

    # bike sharing specifics
    parser.add_argument('--bike_test_season', type=int, default= 1, help='what season to test our model')
    parser.add_argument('--bike_year', type=int, default= 0, help='what year to test our model')

    # misc
    parser.add_argument('--print_base_graph', action='store_true', help='whether to print base classifer comparision graph, can only be used in 1 dimension')
    parser.add_argument('--verbose', action='store_true', help='verbose or not')
    parser.add_argument('--cvs_dir', type=str, default= "./test.cvs", help='path to the cvs file')
    parser.add_argument('--hyper_param_tuning', action='store_true', help='whether to do hyper-parameter tuning')
    parser.add_argument('--data_dir', type=str, default= "data", help='where to put data')
    args = parser.parse_args()
    env = Camelyon17(args)
