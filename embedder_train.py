from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

from SmilesPE.pretokenizer import atomwise_tokenizer

# my files
from validation import general_validation
from embedder import save_checkpoint_embedder
from common_utils import set_seed


# this module params
###################################################
batch_size = 32

# learning rate
lr_initial = 0.003
lr_end = 3 * 1e-4
lr_n_period = 10
lr_n_mult = 1

# KL
kl_start = 0
kl_w_start = 0.0
kl_w_end = 1.0

# gradient clipping during gradient updating
clip_grad = 50
###################################################

# main train function
def fit(args, model, epochs, boundaries, is_validation = False):

    if epochs == 0:
        return
    # data loaders
    train_loader = get_dataloader(model, args, model.dataset.trainset, shuffle=True)

    # optimizer
    optimizer = optim.Adam(get_model_train_params(model), lr=lr_initial)

    # annealers
    lr_annealer = CosineAnnealingLRWithRestart(optimizer)
    kl_annealer = KL_Annealer(epochs)

    # zero gradient
    model.zero_grad()

    if model.name is 'CDN':
        best_criterion = None

    for epoch in range(1, epochs+1):   # [1,2,...,epochs]
        print(' ')
        print('epoch #' + str(epoch))
        kl_weight = kl_annealer(epoch)

        # train 1 epoch
        for i, input_batch in enumerate(train_loader):
            input_batch = tuple(input.to(model.device) for input in input_batch)

            # forward
            kl_loss, recon_loss = model(input_batch)
            loss = kl_weight * kl_loss + recon_loss
            # print('KL_loss=' + str(kl_loss.tolist()) + ', recon_loss=' + str(recon_loss.tolist()))

            # backward
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(get_model_train_params(model), clip_grad)
            optimizer.step()

        # update learning rate
        lr_annealer.step()

        # run validation
        if is_validation is True and (epoch == 1 or epoch % args.validation_freq == 0):
            if epoch == 1:
                fig, ax = plt.subplots()
            avg_similarity, avg_property, avg_SR, avg_validity, avg_novelty, avg_diversity = \
                validation(args, model.name, model, boundaries, epoch, fig=fig, ax=ax)
            if args.plot_results is True:
                fig.savefig(args.plots_folder + '/' + args.property + '/' + model.name + ' validation')

            if model.name is 'CDN':
                # save checkpoint
                current_criterion = avg_similarity
                best_criterion = save_checkpoint_embedder(current_criterion, best_criterion, model, args)
    return model


# validation function
def validation(args, model_name, model, boundaries, epoch, fig=None, ax=None):
    # evaluation mode
    model.eval()

    # dataset loader
    valid_loader = get_dataloader(model, args, model.dataset.validset, batch_size=args.validation_batch_size, shuffle=False)

    # number samples in validset
    validset_len = len(model.dataset.validset)

    # tensor to molecule smiles
    def input_tensor2string(input_tensor):
        return model.tensor2string(input_tensor)

    # generate output molecule from input molecule
    def input2output(input_batch):
        # prepare input
        input_batch = tuple(data.to(model.device) for data in input_batch)

        # embedder encode
        input_batch_emb, _ = model.forward_encoder(input_batch)

        # embedder decode (decode test = input is <bos> and multi for next char + embedding)
        output_batch = model.decoder_test(max_len=args.validation_max_len, embedding=input_batch_emb)

        return output_batch

    trainset = set(model.dataset.trainset)

    # use general validation function
    avg_similarity, avg_property, avg_SR, avg_validity, avg_novelty, avg_diversity = \
        general_validation(args, input2output, input_tensor2string,
        boundaries, valid_loader, validset_len, model_name, trainset, epoch, fig=fig, ax=ax)

    # back to train mode
    model.train()

    return avg_similarity, avg_property, avg_SR, avg_validity, avg_novelty, avg_diversity


# create data loader
def get_dataloader(model, args, data, batch_size=batch_size, collate_fn=None, shuffle=True):
    if collate_fn is None:
        def collate_fn(train_data):
            if model.dataset.use_atom_tokenizer:
                train_data.sort(key=lambda string: len(atomwise_tokenizer(string)), reverse=True)
            else:
                train_data.sort(key=len, reverse=True)
            tensors = [model.dataset.string2tensor(string, model.dataset.c2i, device=model.device) for string in
                       train_data]
            return tensors
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0, worker_init_fn=set_seed(args.seed))


# get training parameters
def get_model_train_params(model):
    return (p for p in model.parameters() if p.requires_grad)


# for KL weights
class KL_Annealer:
    def __init__(self, epochs):
        self.i_start = kl_start
        self.w_start = kl_w_start
        self.w_max = kl_w_end
        self.n_epoch = epochs

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc


# for UGMMT learning rate
class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self, optimizer):
        self.n_period = lr_n_period
        self.n_mult = lr_n_mult
        self.lr_end = lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        super().__init__(optimizer, -1)

    def get_lr(self):
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end


# generate output molecule from input molecule
def input2output_embedder(args, input_batch, model, random_seed_list=None, max_out_len=90, recover_seed=True):
    # prepare input
    input_batch = tuple(data.to(model.device) for data in input_batch)

    random_seed_list = args.seed if random_seed_list is None else random_seed_list
    output_batch = []
    for seed in random_seed_list:
        # set seed
        set_seed(seed)

        # embedder encode
        input_batch_emb, _ = model.forward_encoder(input_batch)

        # embedder decode (decode test = input is <bos> and multi for next char + embedding)
        output_batch += model.decoder_test(max_len=max_out_len, embedding=input_batch_emb)

    if recover_seed is True:
        set_seed(args.seed)
    return output_batch