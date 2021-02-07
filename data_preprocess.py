import pandas as pd
import torch
import numpy as np
from collections import OrderedDict

from SmilesPE.pretokenizer import atomwise_tokenizer

# my files
from property_handler import property_init, property_calc, canonicalize_smiles

# property - DRUG LIKENESS (QED)
QED_dataset_path = 'dataset/QED/'
QED_dataset_filename = 'QED_DATASET.txt'
QED_valid = 800
QED_high_A = 0.78
QED_low_B = 0.91

# property - DRD2
DRD2_dataset_path = 'dataset/DRD2/'
DRD2_dataset_filename = 'DRD2_DATASET.txt'
DRD2_valid = 800
DRD2_high_A = 0.02
DRD2_low_B = 0.85



def create_dataset(property, rebuild_dataset):
    if property is 'QED':
        dataset_path = QED_dataset_path
        dataset_filename = QED_dataset_filename
        valid_size = QED_valid
        high_A = QED_high_A
        low_B = QED_low_B
    elif property is 'DRD2':
        dataset_path = DRD2_dataset_path
        dataset_filename = DRD2_dataset_filename
        valid_size = DRD2_valid
        high_A = DRD2_high_A
        low_B = DRD2_low_B

    if rebuild_dataset is True:
        property_init(property)
        data = pd.read_csv(dataset_path + dataset_filename, header=None)
        paired_list = (data.squeeze()).astype(str).tolist()
        upaired_list = [smiles for pair in paired_list for smiles in pair.split()]
        upaired_list_no_dup = list(OrderedDict.fromkeys(upaired_list))
        dataA, dataB, merged_train_set_property = [], [], []
        for i, mol_smiles in enumerate(upaired_list_no_dup):
            property_val = property_calc(mol_smiles, property)
            merged_train_set_property.append(property_val)
            if property_val <= high_A:
                dataA.append(mol_smiles)
            elif property_val >= low_B:
                dataB.append(mol_smiles)
        A_len = len(dataA)
        B_len = len(dataB)
        print('Total train + validation: ')
        print('Domain A: ' + str(A_len))
        print('Domain B: ' + str(B_len))

        train_size = B_len
        assert(A_len >= train_size + valid_size), "A_len >= train_size + valid_size"

        A_train, _, _ = create_dataset_files(dataA, dataset_path + 'A', train_size, valid_size, 0)
        B_train, _, _ = create_dataset_files(dataB, dataset_path + 'B', train_size, 0, 0)

        ###########  merged train set for CDN and JTVAE training ################
        merged_train = A_train+B_train
        pd.DataFrame(merged_train).to_csv(dataset_path + property+'_mergedAB_specific_train.txt', header=False, index=False)
        merged_train_mol_and_property = []
        for i, mol_smiles in enumerate(merged_train):
            mol_prop = property_calc(mol_smiles, property)
            merged_train_mol_and_property.append(mol_prop)
        pd.DataFrame(merged_train_mol_and_property).to_csv(dataset_path + property + '_mergedAB_specific_property.txt', header=False,
                                              index=False)
        ########################################################################

        ########### valid: "mol + score" for JTVAE optimization ################
        valid_ds = pd.read_csv(dataset_path + 'g2g_validation.txt', header=None)
        valid_set = (valid_ds.squeeze()).astype(str).tolist()
        valid_set_mol_and_property = []
        for i, mol_smiles in enumerate(valid_set):
            property_val = property_calc(mol_smiles, property)
            valid_set_mol_and_property.append(mol_smiles + ' ' + str(property_val))
        pd.DataFrame(valid_set_mol_and_property).to_csv(dataset_path + property + '_valid_mol_property.txt', header=False,
                                              index=False)
        ########################################################################

        ########### test: "mol + score" for JTVAE optimization ################
        test_ds = pd.read_csv(dataset_path + 'A_test.txt', header=None)
        test_set = (test_ds.squeeze()).astype(str).tolist()
        test_set_mol_and_property = []
        for i, mol_smiles in enumerate(test_set):
            property_val = property_calc(mol_smiles, property)
            test_set_mol_and_property.append(mol_smiles + ' ' + str(property_val))
        pd.DataFrame(test_set_mol_and_property).to_csv(dataset_path + property + '_test_mol_property.txt', header=False,
                                              index=False)
        ########################################################################

    # declare domains boundaries
    boundaries = Boundary(high_A, low_B)
    return dataset_path + 'A', dataset_path + 'B', boundaries


def create_dataset_files(data, out_file_path, trainset_size, validset_size=None, testset_size=None):
    train_valid_test_set = np.random.choice(a=data, size=trainset_size + validset_size + testset_size, replace=False).tolist()

    # train
    if trainset_size > 0:
        train_set = train_valid_test_set[:trainset_size]
        pd.DataFrame(train_set).to_csv(out_file_path + '_train.txt', header=False, index=False)

    # validation
    if validset_size > 0:
        valid_set = train_valid_test_set[trainset_size:trainset_size+validset_size]
        pd.DataFrame(valid_set).to_csv(out_file_path + '_validation.txt', header=False, index=False)
    else:
        valid_set = None

    # test
    if testset_size > 0:
        test_set = train_valid_test_set[trainset_size+validset_size:trainset_size+validset_size+testset_size]
        pd.DataFrame(test_set).to_csv(out_file_path + '_test.txt', header=False, index=False)
    else:
        test_set = None

    return train_set, valid_set, test_set


# for holding domains boundaries
class Boundary(object):
    A_boundary = None
    B_boundary = None
    middle = None
    def __init__(self, A_boundary, B_boundary):
        self.A_boundary = A_boundary
        self.B_boundary = B_boundary
        self.middle = (A_boundary + B_boundary)/2
    def get_boundary(self, domain):
        if domain is 'A':
            return self.A_boundary
        elif domain is 'B':
            return self.B_boundary
        else:
            return self.middle


# for holding datasets, 1 for A and 1 for B
class Dataset(object):
    trainset = None
    validset = None
    vocab = None
    c2i = None
    i2c = None
    use_atom_tokenizer = False

    def __init__(self, filename, use_atom_tokenizer=False, isB=False):
        self.use_atom_tokenizer = use_atom_tokenizer
        self.trainset = pd.read_csv(filename + '_train.txt', header=None).iloc[:,0].tolist()
        if isB:
            data = self.trainset
        else:
            self.validset = pd.read_csv(filename + '_validation.txt', header=None).iloc[:,0].tolist()
            data = self.trainset + self.validset

        chars = set()
        if self.use_atom_tokenizer:
            for string in data:
                chars.update(set(atomwise_tokenizer(string)))
        else:
            for string in data:
                chars.update(string)
        all_sys = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']
        self.vocab = all_sys
        self.c2i = {c: i for i, c in enumerate(all_sys)}
        self.i2c = {i: c for i, c in enumerate(all_sys)}

    def in_trainset(self, string):
        return string in self.trainset


    def char2id(self, char, c2i):
        return c2i['<unk>'] if char not in c2i else c2i[char]

    def id2char(self, id, i2c):
        return '<unk>' if id not in i2c else i2c[id]

    def string2ids(self, string, c2i, add_bos=False, add_eos=False):
        if self.use_atom_tokenizer:
            string = atomwise_tokenizer(string)
        ids = [self.char2id(c, c2i) for c in string]
        if add_bos:
            ids = [c2i['<bos>']] + ids
        if add_eos:
            ids = ids + [c2i['<eos>']]
        return ids
    def ids2string(self, ids, c2i, i2c, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == c2i['<bos>']:
            ids = ids[1:]
        if rem_eos and ids[-1] == c2i['<eos>']:
            ids = ids[:-1]
        string = ''.join([self.id2char(id, i2c) for id in ids])
        return string
    def string2tensor(self, string, c2i, device='model'):
        ids = self.string2ids(string, c2i, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long,device=device if device == 'model' else device)
        return tensor

def filname2testset(testset_filename, model_in, model_out, drugs=False):
    print(' ')

    # create testset loader from file
    if drugs:
        df = pd.read_csv(testset_filename)
        testset = set(df['smiles'].dropna())
    else:
        df = pd.read_csv(testset_filename, header=None)
        testset = set(df.iloc[:,0])
    print('Initial number of molecules: ' + str(len(testset)))

    canonicalized_testset = []
    for sample in testset:
        try:
            canonicalized_sample = canonicalize_smiles(sample)
            canonicalized_testset.append(canonicalized_sample)
        except:
            continue
    print('Not Nan canonicalized molecules: ' + str(len(canonicalized_testset)))

    # delete all strings containing chars not from vocab
    vocab_set = set(model_in.dataset.vocab)
    if model_in.dataset.use_atom_tokenizer:
        testset_restricted = [mol for mol in canonicalized_testset if set(atomwise_tokenizer(mol)).issubset(vocab_set)]
    else:
        testset_restricted = [mol for mol in canonicalized_testset if set(mol).issubset(vocab_set)]
    print('With compatible vocabulary: ' + str(len(testset_restricted)))

    # delete and report overlapping molecules in testset and train+validation sets
    if model_in.dataset.validset is None:
        model_in_set = set(model_in.dataset.trainset)
    else:
        model_in_set = set(model_in.dataset.trainset).union(set(model_in.dataset.validset))
    if model_out.dataset.validset is None:
        model_out_set = set(model_out.dataset.trainset)
    else:
        model_out_set = set(model_out.dataset.trainset).union(set(model_out.dataset.validset))
    test_train_and_valid_inter = set(testset_restricted).intersection(model_in_set.union(model_out_set))
    testset_restricted = testset_restricted if test_train_and_valid_inter == set() else list(set(testset_restricted) - test_train_and_valid_inter)
    print('After (Test) and (Train & Validation) intersection removal: ' + str(len(testset_restricted)))

    # check all characters in the testset have appeared in the train + validation sets
    chars = set()
    if model_in.dataset.use_atom_tokenizer:
        for string in testset_restricted:
            chars.update(set(atomwise_tokenizer(string)))
    else:
        for string in testset_restricted:
            chars.update(string)
    test_vocab = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']
    testset_restricted.sort()

    if drugs:
        df_smiles_name = df.loc[df['smiles'].isin(testset_restricted)][['Name', 'smiles']]
        smiles_name_dict = df_smiles_name.set_index('smiles').to_dict()['Name']
        return testset_restricted, smiles_name_dict if set(test_vocab).issubset(vocab_set) else exit()

    return testset_restricted if set(test_vocab).issubset(vocab_set) else exit()