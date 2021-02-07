import argparse
import torch
import pandas as pd


# my files
from common_utils import set_seed, valid_results_file_to_metrics, process_results_file
from property_handler import rdkit_no_error_print, property_init

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Unpaired Generative Molecule-to-Molecule Translator'
    )
    # end-end model settings
    parser.add_argument('--check_testset', default=True, action='store_true', help='get test results')
    parser.add_argument('--property', type=str, default='QED', help='name of property to translate (should be folder with that name inside dataset)')
    parser.add_argument('--baseline', type=str, default='UGMMT', help='testset filename')
    parser.add_argument('--SR_similarity', type=int, default=0.3, help='minimal similarity for success')
    parser.add_argument('--SR_property_val', type=int, default=0.8, help='minimal property value for success')
    parser.add_argument('--seed', type=int, default=50, help='base seed')

    args = parser.parse_args()
    return args


# check testset
def check_testset(args):
    results_file_path = 'baselines_outputs/' + args.property + '/' + args.baseline + '_' + args.property + '.txt'
    valid_results_file_path = 'baselines_outputs/' + args.property + '/valid_' + args.baseline + '_' + args.property + '.txt'
    print(' ')
    print('Loading results for model => ' + args.baseline)
    print('From file => ' + results_file_path)

    # testset len for validaity
    if args.property is 'DRD2':
        testset_len = 1000
    elif args.property is 'QED':
        testset_len = 800

    # train set for novelty
    if args.baseline is 'G2G' or args.baseline is 'CORE':
        train_set_file = 'dataset/' + args.property + '/' + args.property + '_DATASET_split_no_dup.txt'
    elif args.baseline is 'JTVAE' or args.baseline is 'molcg' or args.baseline is 'UGMMT':
        train_set_file = 'dataset/' + args.property + '/' + args.property + '_mergedAB_specific_train.txt'
    else:
        print('unsupported model')
    trainset_df = pd.read_csv(train_set_file, header=None)
    trainset = set((trainset_df.squeeze()).astype(str))

    # result file -> valid results + property and similarity for output molecules
    process_results_file(results_file_path, args, valid_results_file_path, trainset)

    # calculate metrics
    validity_mean, validity_std, \
    diversity_mean, diversity_std, \
    novelty_mean, novelty_std, \
    property_mean, property_std, \
    similarity_mean, similarity_std, \
    SR_mean, SR_std = \
                        valid_results_file_to_metrics(valid_results_file_path, args, testset_len)

    # print results
    print(' ')
    print('Property => ' + args.property)
    print('Baseline => ' + args.baseline)
    print('property => mean: ' + str(round(property_mean, 3)) + '   std: ' + str(round(property_std, 3)))
    print('fingerprint Similarity => mean: ' + str(round(similarity_mean, 3)) + '   std: ' + str(round(similarity_std, 3)))
    print('SR => mean: ' + str(round(SR_mean, 3)) + '   std: ' + str(round(SR_std, 3)))
    print('validity => mean: ' + str(round(validity_mean, 3)) + '   std: ' + str(round(validity_std, 3)))
    print('novelty => mean: ' + str(round(novelty_mean, 3)) + '   std: ' + str(round(novelty_std, 3)))
    print('diversity => mean: ' + str(round(diversity_mean, 3)) + '   std: ' + str(round(diversity_std, 3)))


if __name__ == "__main__":

    with torch.no_grad():
        # parse arguments
        args = parse_arguments()

        # set seed
        set_seed(args.seed)

        # set device (CPU/GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        # initialize property value predictor
        property_init(args.property)

        # disable rdkit error messages
        rdkit_no_error_print()

        # check testset
        if args.check_testset is True:
            check_testset(args)
