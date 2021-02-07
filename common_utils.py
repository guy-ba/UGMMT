import torch
import numpy as np
import random
import pandas as pd

# my files
from property_handler import property_calc, similarity_calc, smiles2fingerprint, is_valid_molecule


# set seed
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    random.SystemRandom(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# generate output molecule from input molecule
def input2output(args, input_batch, model_in, T, model_out, random_seed_list=None, max_out_len=90, recover_seed=True):
    # prepare input
    input_batch = tuple(data.to(model_in.device) for data in input_batch)

    if args.use_fp and args.use_EETN:
        # prepare finger prints
        input_batch_fp_str = [smiles2fingerprint(model_in.tensor2string(input), fp_translator=True) for input in input_batch]
        input_batch_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in input_batch_fp_str]).to(
            model_in.device)
        input_batch_fp = input_batch_fp.detach()
    else:
        input_batch_fp = None

    random_seed_list = args.seed if random_seed_list is None else random_seed_list
    output_batch = []
    for seed in random_seed_list:
        # set seed
        set_seed(seed)
        if args.conditional:
            translated_batch_emb = input_batch_fp
        else:
            # embedder encode (METN)
            input_batch_emb, _ = model_in.forward_encoder(input_batch)

            if args.use_EETN:
                # embedding translator (EETN)
                translated_batch_emb = T(input_batch_emb, input_batch_fp)
            else:
                translated_batch_emb = input_batch_emb

        # embedder decode (decode test = input is <bos> and multi for next char + embedding) (METN)
        output_batch += model_out.decoder_test(max_len=max_out_len, embedding=translated_batch_emb)

    if recover_seed is True:
        set_seed(args.seed)
    return output_batch



# generate intermediate embeddings and output from input
def input2all(args, input_batch, model_in, T, model_out, max_out_len=90):
    # prepare input
    input_batch = tuple(data.to(model_in.device) for data in input_batch)

    if args.use_fp and args.use_EETN:
        # prepare finger prints
        input_batch_fp_str = [smiles2fingerprint(model_in.tensor2string(input), fp_translator=True) for input in input_batch]
        input_batch_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in input_batch_fp_str]).to(
            model_in.device)
        input_batch_fp = input_batch_fp.detach()
    else:
        input_batch_fp = None

    # embedder encode (METN)
    input_batch_emb, _ = model_in.forward_encoder(input_batch)

    if args.use_EETN:
        # embedding translator (EETN)
        translated_batch_emb = T(input_batch_emb, input_batch_fp)
    else:
        translated_batch_emb = input_batch_emb

    # embedder decode (decode test = input is <bos> and multi for next char + embedding) (METN)
    output_batch = model_out.decoder_test(max_len=max_out_len, embedding=translated_batch_emb)

    return input_batch_emb, translated_batch_emb, output_batch


# check if output molecule is novel (different form the input molecule and not in trainset)
def is_novel(input_mol_smiles, output_mol_smiles, trainset):
    return (input_mol_smiles != output_mol_smiles) and (output_mol_smiles not in trainset)


# generate a list with length random integers, each one from [0,100000]
def get_random_list(length, last=100000):
    return [random.randint(0, last) for i in range(length)]


def generate_results_file(test_loader, input2output_func, input2smiles, results_file_path):
    in_mols, out_mols = [], []
    for i, input_batch in enumerate(test_loader):
        current_batch_size = len(input_batch)

        # generate output batch
        output_batch = input2output_func(input_batch)

        # for every input molecule
        for j, input in enumerate(input_batch):
            input_molecule_smiles = input2smiles(input)  # to smiles
            output_molecule_smiles_list = output_batch[j::current_batch_size]

            out_mols.extend(output_molecule_smiles_list)
            in_mols.extend([input_molecule_smiles] * len(output_molecule_smiles_list))

    results = pd.DataFrame(list(zip(in_mols, out_mols)))
    results.to_csv(results_file_path, index=False, header=False, sep=' ')


def process_results_file(res_file_path, args, valid_res_file_path, trainset):
    res_df = pd.read_csv(res_file_path, header=None, delimiter=' ')
    res_df.rename(columns={0: 'input', 1: 'output'}, inplace=True)

    # filter valid molecules
    valid_res_def = res_df[np.vectorize(is_valid_molecule)(res_df['output'], args.property)]

    if valid_res_def.empty:
        valid_res_def = pd.DataFrame(columns=['input', 'output', args.property, 'sim', 'novel'])
        valid_res_def = valid_res_def.append({'input':'invalid', 'output':'invalid', args.property:0, 'sim':0, 'novel':0}, ignore_index=True)
    else:
        # add property column for output molecules
        valid_res_def[args.property] = np.vectorize(property_calc)(valid_res_def['output'], args.property)

        # add similarity between input and output molecules column
        valid_res_def['sim'] = np.vectorize(similarity_calc)(valid_res_def['input'], valid_res_def['output'])

        # add novelty column for output molecules
        valid_res_def['novel'] = np.vectorize(is_novel)(valid_res_def['input'], valid_res_def['output'], trainset)

    # save output df
    valid_res_def.to_csv(valid_res_file_path, index=False)


# calculate metrics
def valid_results_file_to_metrics(valid_res_file_path, args, num_source_mols):
    valid_df = pd.read_csv(valid_res_file_path)

    validity, diversity, novelty, property, similarity, SR = [], [], [], [], [], []
    for retry_i in range(10):
        validity_mean, diversity_mean, novelty_mean, property_mean, similarity_mean, SR_mean = \
        get_metics_for_sample(valid_df, args, how_many_samples=1, num_source_mols=num_source_mols, seed=retry_i)

        validity.append(validity_mean)
        diversity.append(diversity_mean)
        novelty.append(novelty_mean)
        property.append(property_mean)
        similarity.append(similarity_mean)
        SR.append(SR_mean)

    validity_np, diversity_np, novelty_np, property_np, similarity_np, SR_np = np.array(validity), \
        np.array(diversity), np.array(novelty), np.array(property), np.array(similarity), np.array(SR)
    return validity_np.mean(), validity_np.std(), \
           diversity_np.mean(), diversity_np.std(), \
           novelty_np.mean(), novelty_np.std(), \
           property_np.mean(), property_np.std(), \
           similarity_np.mean(), similarity_np.std(), \
           SR_np.mean(), SR_np.std()

def get_metics_for_sample(valid_df, args, how_many_samples, num_source_mols, seed):
    # shuffle
    valid_df = valid_df.sample(len(valid_df), replace=False, random_state=seed)

    # get sample for each source molecule
    valid_df = valid_df.groupby(['input'], as_index=False).head(how_many_samples)
    num_valid_mols = len(valid_df)

    # get unique output molecules
    unique_out_mols = valid_df['output'].unique()

    # calculate *** VALIDITY ***
    validity = num_valid_mols / num_source_mols

    # calculate *** DIVERSITY ***
    diversity = len(unique_out_mols) / num_valid_mols

    # calculate *** SR ***
    SR = len(valid_df[(valid_df['sim'] > args.SR_similarity) & (valid_df[args.property] > args.SR_property_val) & (valid_df['novel'])]) / num_valid_mols

    return validity, diversity, valid_df['novel'].mean(), valid_df[args.property].mean(), valid_df['sim'].mean(), SR