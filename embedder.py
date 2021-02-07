import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Encoder
encoder_hid_dim = 256
encoder_num_layers = 1
encoder_bidir = True
encoder_dropout = 0.5

# Embedding
embedding_dim = 256

# Decoder
decoder_num_layers = 3
decoder_dropout = 0
decoder_hid_dim = 512


class Embedder(nn.Module):
    def __init__(self, dataset, name, args):
        super().__init__()
        self.name = name
        self.dataset = dataset
        final_embedding_dim = 2048 if args.conditional else embedding_dim   # for conditional ablation

        vocab_size_and_dim = len(dataset.vocab)     #it's also character embedding dim
        self.char_embedder = nn.Embedding(num_embeddings=vocab_size_and_dim, embedding_dim=vocab_size_and_dim, padding_idx=dataset.c2i['<pad>'])
        self.char_embedder.weight.data.copy_(torch.eye(vocab_size_and_dim))

        # Encoder
        self.encoder_GRU = nn.GRU(input_size=vocab_size_and_dim, hidden_size=encoder_hid_dim, num_layers=encoder_num_layers, batch_first=True,
                                  dropout=encoder_dropout if encoder_num_layers > 1 else 0, bidirectional=encoder_bidir)
        encoder_GRU_out_dim = encoder_hid_dim * (2 if encoder_bidir else 1)
        self.encoder_mu = nn.Linear(encoder_GRU_out_dim, final_embedding_dim)
        self.encoder_logVar = nn.Linear(encoder_GRU_out_dim, final_embedding_dim)

        # Decoder
        self.decoder_emb_to_hid = nn.Linear(final_embedding_dim, decoder_hid_dim)
        self.decoder_GRU = nn.GRU(input_size=vocab_size_and_dim + final_embedding_dim, hidden_size=decoder_hid_dim, num_layers=decoder_num_layers,
                                  batch_first=True, dropout=decoder_dropout if decoder_num_layers > 1 else 0)
        self.decoder_GRU_out_to_char_score = nn.Linear(decoder_hid_dim, vocab_size_and_dim)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([self.encoder_GRU, self.encoder_mu, self.encoder_logVar])
        self.decoder = nn.ModuleList([self.decoder_emb_to_hid, self.decoder_GRU, self.decoder_GRU_out_to_char_score])
        self.embedder = nn.ModuleList([self.char_embedder, self.encoder, self.decoder])

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = self.dataset.string2ids(string, self.dataset.c2i, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device if device == 'model' else device)
        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.dataset.ids2string(ids, self.dataset.c2i, self.dataset.i2c, rem_bos=True, rem_eos=True)
        return string

    def forward(self, tuple_strings):
        embedding, KL_loss = self.forward_encoder(tuple_strings)
        recon_loss, batch_strings_recon_score = self.forward_decoder(tuple_strings, embedding)
        return KL_loss, recon_loss

    def forward_encoder(self, tuple_strings):
        # embed strings and apply GRU encoder
        list_strings_emb = [self.char_embedder(string) for string in tuple_strings]
        strings_emb = nn.utils.rnn.pack_sequence(list_strings_emb)
        _, h = self.encoder_GRU(strings_emb, None)

        # handle bi-directional: concatenate directions
        h = h[-(1 + int(self.encoder_GRU.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        # reparametrization trick
        mu, logVar = self.encoder_mu(h), self.encoder_logVar(h)
        eps = torch.randn_like(mu)
        embedding = mu + (logVar / 2).exp() * eps
        KL_loss = 0.5 * (logVar.exp() + mu ** 2 - 1 - logVar).sum(1).mean()
        return embedding, KL_loss


    def forward_decoder(self, tuple_strings, embedding):
        # prepare initial hidden state for GRU
        h_0 = self.decoder_emb_to_hid(embedding)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_GRU.num_layers, 1, 1)

        # prepare input to GRU
        lengths = [len(string) for string in tuple_strings]
        strings = nn.utils.rnn.pad_sequence(tuple_strings, batch_first=True, padding_value=self.dataset.c2i['<pad>'])
        strings_emb = self.char_embedder(strings)
        embedding_replica = embedding.unsqueeze(1).repeat(1, strings_emb.size(1), 1)
        GRU_input = torch.cat([strings_emb, embedding_replica], dim=-1)
        GRU_input = nn.utils.rnn.pack_padded_sequence(GRU_input, lengths, batch_first=True)

        # apply GRU decoder
        GRU_output, _ = self.decoder_GRU(GRU_input, h_0)
        GRU_output, _ = nn.utils.rnn.pad_packed_sequence(GRU_output, batch_first=True)

        # strings reconstruction and reconstruction loss
        strings_recon_score = self.decoder_GRU_out_to_char_score(GRU_output)
        recon_loss = 10 * F.cross_entropy(strings_recon_score[:, :-1].contiguous().view(-1, strings_recon_score.size(-1)),
                                          strings[:, 1:].contiguous().view(-1),
                                          ignore_index=self.dataset.c2i['<pad>'])
        return recon_loss, strings_recon_score


    def decoder_test(self, max_len=100, embedding=None, temp=1.0):
        with torch.no_grad():
            batch_size = embedding.shape[0]
            if embedding is None:
                embedding = torch.randn(batch_size, self.encoder_mu.out_features, device=self.char_embedder.weight.device)
            else:
                # prepare hidden state for GRU
                embedding = embedding.to(self.device)
                embedding_for_GRU_input = embedding.unsqueeze(1)
                h = self.decoder_emb_to_hid(embedding)
                h = h.unsqueeze(0).repeat(self.decoder_GRU.num_layers, 1, 1)

                # initialize initial char and output
                next_char = torch.tensor(self.dataset.c2i['<bos>'], device=self.device).repeat(batch_size)
                output = torch.tensor([self.dataset.c2i['<pad>']], device=self.device).repeat(batch_size, max_len)
                output[:, 0] = self.dataset.c2i['<bos>']

                # last char index and mask for outputs
                end_pads = torch.tensor([max_len], device=self.device).repeat(batch_size)
                eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

                for i in range(1, max_len):
                    # run decoder GRU to generate next char
                    next_char_emb = self.char_embedder(next_char).unsqueeze(1)
                    GRU_input = torch.cat([next_char_emb, embedding_for_GRU_input], dim=-1) #input = next char + embedding
                    GRU_output, h = self.decoder_GRU(GRU_input, h)
                    strings_recon_score = self.decoder_GRU_out_to_char_score(GRU_output.squeeze(1))
                    strings_recon_prob = F.softmax(strings_recon_score / temp, dim=-1)

                    # sample next char from multinomial distribution
                    next_char = torch.multinomial(strings_recon_prob, 1)[:, 0]

                    # update output if output did not end
                    output[~eos_mask, i] = next_char[~eos_mask]

                    # update last char index and mask for outputs
                    i_eos_mask = (~eos_mask).type(torch.bool) & (next_char == self.dataset.c2i['<eos>']).type(torch.bool)
                    end_pads[i_eos_mask] = i + 1
                    eos_mask = eos_mask | i_eos_mask

                    # copy every char update
                    final_output = []
                    for i in range(output.size(0)):
                        final_output.append(output[i, :end_pads[i]])

        return [self.tensor2string(i_x) for i_x in final_output]


def save_checkpoint_embedder(current_criterion, best_criterion, model, args):
    # first model or best model so far
    if best_criterion is None or current_criterion > best_criterion:
        best_criterion = current_criterion
        saved_state = dict(best_criterion=best_criterion,
                           model_embedder=model)
        checkpoint_filename_path = args.checkpoints_folder + '/' + args.property + '/CDN/checkpoint_model.pth'
        torch.save(saved_state, checkpoint_filename_path)
        print('*** Saved checkpoint in: ' + checkpoint_filename_path + ' ***')
    return best_criterion


def load_checkpoint_embedder(args, device):
    checkpoint_filename_path = args.checkpoints_folder + '/' + args.property + '/CDN/checkpoint_model.pth'
    if os.path.isfile(checkpoint_filename_path):
        print('*** Loading checkpoint file ' + checkpoint_filename_path)
        saved_state = torch.load(checkpoint_filename_path,
                                 map_location=device)

        best_criterion = saved_state.get('best_criterion')
        model = saved_state['model_embedder']
    return model, best_criterion