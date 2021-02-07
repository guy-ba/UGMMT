import argparse
import torch
import itertools
import matplotlib.pyplot as plt

# my files
from data_preprocess import create_dataset, Dataset
from embedder import Embedder
from embedder_train import fit, get_model_train_params, get_dataloader
from embedding_translator import Translator, Discriminator, weights_init_normal, LambdaLR, ReplayBuffer, Statistics, save_checkpoint
from property_handler import smiles2fingerprint, rdkit_no_error_print
from validation import general_validation
from common_utils import set_seed, input2output, get_random_list


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Unpaired Generative Molecule-to-Molecule Translator'
    )

    # end-end model settings
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train the end-end model')
    parser.add_argument('--epoch_init', type=int, default=1, help='initial epoch')
    parser.add_argument('--epoch_decay', type=int, default=90, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for end-end model')
    parser.add_argument('--property', type=str, default='QED', help='name of property to translate (should be folder with that name inside dataset)')
    parser.add_argument('--init_lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--is_valid', default=True, action='store_true', help='run validation every train epoch')
    parser.add_argument('--valid_direction', type=str, default='AB', help='direction of validation translation- AB: A->B; BA: B->A')
    parser.add_argument('--plot_results', default=True, action='store_true', help='plot validation set results during end-end model training')
    parser.add_argument('--print_results', default=True, action='store_true', help='print validation results during end-end model training')
    parser.add_argument('--rebuild_dataset', default=False, action='store_false', help='rebuild dataset files')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints', help='name of folder for checkpoints saving')
    parser.add_argument('--plots_folder', type=str, default='plots_output', help='name of folder for plots saving')
    parser.add_argument('--early_stopping', type=int, default=15, help='Whether to stop training early if there is no\
                        criterion improvement for this number of validation runs.')
    parser.add_argument('--seed', type=int, default=50, help='base seed')
    parser.add_argument('--num_retries', type=int, default=20, help='number of retries for each validation sample')
    parser.add_argument('--SR_similarity', type=int, default=0.3, help='minimal similarity for success')
    parser.add_argument('--SR_property_val', type=int, default=0.8, help='minimal property value for success')
    parser.add_argument('--validation_max_len', type=int, default=90, help='length of validation smiles')
    parser.add_argument('--validation_batch_size', type=int, default=32, help='batch size for validation end-end model')
    parser.add_argument('--validation_freq', type=int, default=3, help='validate every n-th epoch')
    parser.add_argument('--is_CDN', default=False, action='store_false', help='trains CDN network')
    parser.add_argument('--tokenize', default=False, action='store_false', help='use atom tokenization')
    parser.add_argument('--cycle_loss', default=True, action='store_true', help='use cycle loss during training or not')

    # Ablation
    parser.add_argument('--gan_loss', default=False, action='store_true', help='use gan loss during training or not')
    parser.add_argument('--kl_loss', default=False, action='store_true', help='use kl loss during training or not')
    parser.add_argument('--swap_cycle_fp', default=False, action='store_true', help='swap fp in second translator during training or not')
    parser.add_argument('--use_fp', default=True, action='store_true', help='does translator use molecule fp')
    parser.add_argument('--use_EETN', default=True, action='store_true', help='use EETN network')
    parser.add_argument('--conditional', default=False, action='store_true', help='using only fp for optimization')
    parser.add_argument('--no_pre_train', default=False, action='store_true', help='disable METNs pre training')



    args = parser.parse_args()
    return args


def train_iteration_T(real_A, real_B, model_A, model_B, T_AB, T_BA, D_A, D_B, loss_GAN, optimizer_T, fake_A_buffer, fake_B_buffer, args):
    optimizer_T.zero_grad()

    if args.conditional:
        real_B_fp_str = [smiles2fingerprint(model_B.tensor2string(mol), fp_translator=True) for mol in real_B]
        real_B_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_B_fp_str]).to(device)
        real_B_fp = real_B_fp.detach()

        loss, _ = model_B.forward_decoder(real_B, real_B_fp)
        loss.backward()
        optimizer_T.step()
        return loss, None, None, None, None, None, None, None, None

    # embedder (METN)
    real_A_emb, kl_loss_A = model_A.forward_encoder(real_A)
    real_B_emb, kl_loss_B = model_B.forward_encoder(real_B)
    if args.kl_loss is False:
        kl_loss_A, kl_loss_B = None, None

    if args.use_EETN is False:                  # for ablation
        recon_loss_A, _ = model_A.forward_decoder(real_A, real_A_emb)
        recon_loss_B, _ = model_B.forward_decoder(real_B, real_B_emb)
        loss = 2 * recon_loss_A + recon_loss_B
        loss.backward()
        optimizer_T.step()
        return loss, None, None, recon_loss_A, recon_loss_B, None, None, None, None

    # prepare fingerprints
    if args.use_fp:
        real_A_fp_str = [smiles2fingerprint(model_A.tensor2string(mol), fp_translator=True) for mol in real_A]
        real_B_fp_str = [smiles2fingerprint(model_B.tensor2string(mol), fp_translator=True) for mol in real_B]
        real_A_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_A_fp_str]).to(device)
        real_B_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_B_fp_str]).to(device)
        real_A_fp = real_A_fp.detach()
        real_B_fp = real_B_fp.detach()
    else:
        real_A_fp, real_B_fp = None, None


    fake_B_emb = T_AB(real_A_emb, real_A_fp)
    fake_A_emb = T_BA(real_B_emb, real_B_fp)

    # GAN (Adversarial) loss for ablation
    if args.gan_loss:
        pred_fake_b = D_B(fake_B_emb)
        loss_GAN_AB = loss_GAN(pred_fake_b, torch.ones(pred_fake_b.size(), device=device, requires_grad=False))

        pred_fake_a = D_A(fake_A_emb)
        loss_GAN_BA = loss_GAN(pred_fake_a, torch.ones(pred_fake_a.size(), device=device, requires_grad=False))
    else:
        loss_GAN_AB, loss_GAN_BA = None, None


    # Cycle loss
    if args.cycle_loss:
        if args.swap_cycle_fp:          # for ablation
            cycle_A_emb = T_BA(fake_B_emb, real_B_fp)
            cycle_B_emb = T_AB(fake_A_emb, real_A_fp)

        else:
            cycle_A_emb = T_BA(fake_B_emb, real_A_fp)
            cycle_B_emb = T_AB(fake_A_emb, real_B_fp)

        cycle_loss_A, _ = model_A.forward_decoder(real_A, cycle_A_emb)
        cycle_loss_B, _ = model_B.forward_decoder(real_B, cycle_B_emb)
    else:
        cycle_loss_A, cycle_loss_B = None, None

    # Total loss
    if args.cycle_loss and args.gan_loss is False and args.kl_loss is False:    # Main model: only cycle
        loss = 2 * cycle_loss_A + cycle_loss_B
    elif args.cycle_loss and args.gan_loss and args.kl_loss:                    # for ablation: cycle + gan + kl (for stability)
        loss = 2 * cycle_loss_A + cycle_loss_B + \
                 loss_GAN_AB + loss_GAN_BA + \
                 0.2 * kl_loss_A + 0.2 * kl_loss_B
    elif args.cycle_loss and args.gan_loss is False and args.kl_loss:           # for ablation: cycle + kl
        loss = 2 * cycle_loss_A + cycle_loss_B + \
                 kl_loss_A + kl_loss_B
    else:
        print('No such setting for the main model, nor for the ablation tests')
        exit()

    loss.backward()
    optimizer_T.step()

    # for discriminators usage (ablation)
    if args.gan_loss:
        fake_A_emb = fake_A_buffer.push_and_pop(fake_A_emb)
        fake_B_emb = fake_B_buffer.push_and_pop(fake_B_emb)
    else:
        fake_A_emb, fake_B_emb = None, None

    return loss, loss_GAN_AB, loss_GAN_BA, cycle_loss_A, cycle_loss_B, kl_loss_A, kl_loss_B, fake_A_emb, fake_B_emb


# for ablation
def train_iteration_D(real, fake_emb, model, D, loss_GAN, optimizer_D):
    optimizer_D.zero_grad()

    real_emb, kl_loss = model.forward_encoder(real)

    # Real loss
    pred_real = D(real_emb.detach())
    loss_D_real = loss_GAN(pred_real, torch.ones(pred_real.size(), device=device, requires_grad=False))

    # Fake loss
    pred_fake = D(fake_emb.detach())
    loss_D_fake = loss_GAN(pred_fake, torch.zeros(pred_fake.size(), device=device, requires_grad=False))

    # Total loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5

    loss_D.backward()
    optimizer_D.step()

    return loss_D, loss_D_real, loss_D_fake


def validation(args, model_name, model_in, model_out, T, epoch, boundaries, random_seed_list, fig=None, ax=None):
    # evaluation mode
    model_in.eval()
    model_out.eval()
    T.eval()

    # dataset loader
    valid_loader = get_dataloader(model_in, args, model_in.dataset.validset, batch_size=args.validation_batch_size, shuffle=False)

    # number samples in validset
    validset_len = len(model_in.dataset.validset)

    # tensor to molecule smiles
    def input_tensor2string(input_tensor):
        return model_in.tensor2string(input_tensor)

    trainset = set(model_in.dataset.trainset).union(model_out.dataset.trainset)

    #generate output molecule from input molecule
    def local_input2output(input_batch):
        return input2output(args, input_batch, model_in, T, model_out, random_seed_list,
                            max_out_len=args.validation_max_len)

    # use general validation function
    avg_similarity, avg_property, avg_SR, avg_validity, avg_novelty, avg_diversity =\
        general_validation(args, local_input2output, input_tensor2string, boundaries, valid_loader, validset_len, model_name, trainset, epoch,
        fig=fig, ax=ax)

    # back to train mode
    model_in.train()
    model_out.train()
    T.train()

    return avg_similarity, avg_property, avg_SR, avg_validity, avg_novelty, avg_diversity

def early_stop(early_stopping, current_criterion, best_criterion, runs_without_improvement):
    if early_stopping is not None:
        # first model or best model so far
        if best_criterion is None or current_criterion > best_criterion:
            runs_without_improvement = 0
        # no improvement
        else:
            runs_without_improvement += 1
        if runs_without_improvement >= early_stopping:
            return True, runs_without_improvement       # True = stop training
        else:
            return False, runs_without_improvement


if __name__ == "__main__":

    # parse arguments
    args = parse_arguments()

    # set seed
    set_seed(args.seed)

    # set device (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # disable rdkit error messages
    rdkit_no_error_print()

    # epochs for METN pre training
    if args.property is 'QED':
        embedder_epochs_num = 1
    elif args.property is 'DRD2':
        embedder_epochs_num = 12
    else:
        print("property must bt 'QED 'or 'DRD2'")
        exit()

    if args.conditional or args.no_pre_train:
            embedder_epochs_num = 0

    if args.is_CDN is True:
        _, _, boundaries = create_dataset(args.property, rebuild_dataset=False)
        dataset_CDN = Dataset('dataset/' + args.property + '/CDN/CDN')
        model_CDN = Embedder(dataset_CDN, 'CDN', args).to(device)
        embedder_epochs_num = args.epochs
        fit(args, model_CDN, embedder_epochs_num, boundaries, is_validation=True)
        exit()

    # prepare dataset
    dataset_file_A, dataset_file_B, boundaries = create_dataset(args.property, args.rebuild_dataset)
    dataset_A = Dataset(dataset_file_A, use_atom_tokenizer=args.tokenize, isB=False)
    dataset_B = Dataset(dataset_file_B, use_atom_tokenizer=args.tokenize, isB=True)

    # create  and pre-train the embedders (METNs)
    model_A = Embedder(dataset_A, 'Embedder A', args).to(device)
    fit(args, model_A, embedder_epochs_num, boundaries, is_validation=True)
    model_B = Embedder(dataset_B, 'Embedder B', args).to(device)
    fit(args, model_B, embedder_epochs_num, boundaries, is_validation=False)

    # create embedding translators (EETN)
    T_AB = Translator().to(device)
    T_BA = Translator().to(device)

    # discriminators for ablation
    if args.gan_loss:
        D_A = Discriminator(1).to(device)
        D_B = Discriminator(1).to(device)
    else:
        D_A, D_B = None, None

    # Adversarial loss for ablation
    if args.gan_loss:
        loss_GAN = torch.nn.MSELoss()
    else:
        loss_GAN = None

    # weights
    T_AB.apply(weights_init_normal)
    T_BA.apply(weights_init_normal)

    # weights for ablation
    if args.gan_loss:
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)


    # optimizer
    optimizer_T = torch.optim.Adam(itertools.chain(T_AB.parameters(), T_BA.parameters(), get_model_train_params(model_A),
                                    get_model_train_params(model_B)), lr=args.init_lr, betas=(0.5, 0.999))
    # optimizers for ablation
    if args.gan_loss:
        optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.init_lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.init_lr, betas=(0.5, 0.999))


    # scheduler
    lr_scheduler_T = torch.optim.lr_scheduler.LambdaLR(optimizer_T, lr_lambda=LambdaLR(args.epochs, args.epoch_init, args.epoch_decay).step)
    # schedulers for ablation
    if args.gan_loss:
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.epochs, args.epoch_init, args.epoch_decay).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.epochs, args.epoch_init, args.epoch_decay).step)


    # train dataloaders
    A_train_loader = get_dataloader(model_A, args, model_A.dataset.trainset, args.batch_size, collate_fn=None, shuffle=True)
    B_train_loader = get_dataloader(model_B, args, model_B.dataset.trainset, args.batch_size, collate_fn=None, shuffle=True)

    # buffer for max_size last fake samples for ablation
    if args.gan_loss:
        fake_A_buffer = ReplayBuffer(max_size=50)
        fake_B_buffer = ReplayBuffer(max_size=50)
    else:
        fake_A_buffer, fake_B_buffer = None, None


    # for early stopping
    best_criterion = None
    runs_without_improvement = 0

    # generate random seeds
    random_seed_list = get_random_list(args.num_retries)

    ###### Training ######
    for epoch in range(args.epoch_init, args.epochs + 1):
        print(' ')
        print('epoch #' + str(epoch))

        # statistics
        stats = Statistics()

        for i, (real_A, real_B) in enumerate(zip(A_train_loader, B_train_loader)):

            # update translators (EETN) and embedders (METNs)
            loss, loss_GAN_AB, loss_GAN_BA, cycle_loss_A, cycle_loss_B, kl_loss_A, kl_loss_B, fake_A_emb, fake_B_emb = \
            train_iteration_T(real_A, real_B, model_A, model_B, T_AB, T_BA, D_A, D_B, loss_GAN,
                              optimizer_T, fake_A_buffer, fake_B_buffer, args)

            # for ablation
            if args.gan_loss:
                # update discriminators
                loss_D_A, loss_D_A_real, loss_D_A_fake = train_iteration_D(real_A, fake_A_emb, model_A, D_A, loss_GAN, optimizer_D_A)
                loss_D_B, loss_D_B_real, loss_D_B_fake = train_iteration_D(real_B, fake_B_emb, model_B, D_B, loss_GAN, optimizer_D_B)
            else:
                loss_D_A, loss_D_A_real, loss_D_A_fake, loss_D_B, loss_D_B_real, loss_D_B_fake = None, None, None, None, None, None

            # update statistics
            stats.update(loss, loss_GAN_AB, loss_GAN_BA, cycle_loss_A, cycle_loss_B, kl_loss_A, kl_loss_B,
                         loss_D_A, loss_D_A_real, loss_D_A_fake, loss_D_B, loss_D_B_real, loss_D_B_fake)

        # print epoch's statistics
        stats.print()

        # run validation
        if args.is_valid is True and (epoch == 1 or epoch % args.validation_freq == 0):
            if args.valid_direction is 'AB' or args.valid_direction is 'Both':
                if epoch == 1:
                    fig_AB, ax_AB = plt.subplots()
                avg_similarity_AB, avg_property_AB, avg_SR_AB, avg_validity_AB, avg_novelty_AB, avg_diversity_AB = \
                    validation(args, 'Our AB', model_A, model_B, T_AB, epoch, boundaries, random_seed_list, fig=fig_AB, ax=ax_AB)
                # save plots
                if args.plot_results is True:
                    fig_AB.savefig(args.plots_folder + '/' + args.property + '/Our AB valid')

            if args.valid_direction is 'BA' or args.valid_direction is 'Both':
                if epoch == 1:
                    fig_BA, ax_BA = plt.subplots()
                avg_similarity_BA, avg_property_BA, avg_SR_BA, avg_validity_BA, avg_novelty_BA, avg_diversity_BA = \
                    validation(args, 'Our BA', model_B, model_A, T_BA, epoch, boundaries, random_seed_list, fig=fig_BA, ax=ax_BA)
                # save plots
                if args.plot_results is True:
                    fig_BA.savefig(args.plots_folder + '/' + args.property + '/Our BA valid')

            # early stopping
            current_criterion = avg_SR_AB
            is_early_stop, runs_without_improvement = \
                early_stop(args.early_stopping, current_criterion, best_criterion, runs_without_improvement)
            if is_early_stop:
                break

            # save checkpoint
            best_criterion = save_checkpoint(current_criterion, best_criterion, T_AB, T_BA, model_A, model_B, args)

        # update learning rate
        lr_scheduler_T.step()

        # for ablation
        if args.gan_loss:
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()
