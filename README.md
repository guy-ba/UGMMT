Unpaired Generative Molecule-to-Molecule Translation code.

We run all training and experiments on Ubuntu 18.04.5 using one Nvidia GeForce RTX 2080 Ti 11GB GPU, two Intel
13 Xeon Gold 6230 2.10GHZ CPUs and 64GB RAM memory.


# Installation:
1. Install conda / minicaonda
2. From the main folder run:\
    i. conda env create -f environment.yml\
    ii. conda activate UGMMT

All dataset files located in dataset folder.

# Training:
From the main folder run:

1. python train.py 2>error_train.txt

train.py is the main training file, contains most of the hyper-parameter and configuration setting.
After training, the checkpoints will be located in the checkpoints folder, training plots will be located in the plots_output folder.

Main setting:\
property (line 27) [property selection] -> QED / DRD2.


# Inference  :
From the main folder run:

1. python test.py 2>error_test.txt

test.py is the main testing file, contains most of the hyper-parameter and configuration setting.

Main setting:\
check_testset (line 23) [Main Results: Molecule Optimization] -> True / False.\
discover_approved_drugs (ine 38) [Optimized Drug Generation] -> True / False.\
property (line 25) [property selection] -> QED / DRD2.


# Ablation Experiments  :
Druing training change gan_loss, kl_loss, swap_cycle_fp, use_fp, use_EETN, conditional and no_pre_train flages (lines 50-56 in train.py) according to:
1. No Pre-train		  -> no_pre_train to True.
2. NO EETN 			    -> use_EETN to False.
3. NO fp 			      -> use_fp to False.
4. Only fp			    -> conditional to True.
5. Swap Cycle fp 	  -> swap_cycle_fp to True.
6. Add Adversarial 	-> gan_loss and kl_loss to True.

For testing, run regularly except for these changes:
1. NO EETN 			-> use_EETN to False (line 50 in test.py).
2. NO fp 			-> use_fp to False (line 49 in test.py).
3. Only fp			-> conditional to True (line 51 in test.py).
