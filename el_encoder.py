import torch
import torch.nn as nn
from el_hyperparams import MAX_POS, TYPE_OPT
from utils import embedding_3D

class ELEncoder(nn.Module):

    def __init__(self, config):
        super(ELEncoder, self).__init__()
        # performs Lp(default p=2) normalization of inputs over specified dimension
        word_embs = nn.functional.normalize(config['word_embs'])
        # creates Embedding instance from given 2-dimensional FloatTensor (num_embeddings and embedding_dim)
        self.word_embs = nn.Embedding.from_pretrained(word_embs, freeze=True)
        # stores embedding's size of position
        self.pos_embs = nn.Embedding(2 * MAX_POS + 1, config['pos_embdim'])
        # stores embedding's size of embedding's type
        self.type_embs = nn.Embedding(config['n_types'], config['type_embdim'])
        # dimension of type embedding
        self.type_embdim = config['type_embdim']
        # dimension of entity embedding
        self.ent_embdim = config['ent_embdim']
        # dimension of bias
        self.rel_embs = nn.Embedding(config['n_rels'], config['ent_embdim'])
        # dimension of weight matrix
        self.rel_weight = nn.Parameter(torch.zeros(config['type_embdim'], config['ent_embdim']))
        # initial the input Tensor using a normal distribution with no gradient will be recorded for this operation
        nn.init.kaiming_normal_(self.rel_weight, mode='fan_out', nonlinearity='relu')
        # dimension after concatenate between the position embedding and the word from GloVe
        dim = word_embs.shape[1] + config['pos_embdim']
        # randomly zeroes some of the elements of the input tensor with probability p = config['dropout']
        self.dropout = nn.Dropout(p=config['dropout'])
        self.type = config.get('type', 'lstm')

        # LSTM encoder
        if self.type == 'lstm':
            self.lstm = nn.LSTM(dim, config['lstm_hiddim'], 2, batch_first=True, bidirectional=True, dropout=config['dropout'])
            en_hiddim1 = 4 * config['lstm_hiddim'] + config['ent_embdim']

        # PCNN encoder
        elif self.type == 'pcnn':
            self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=config['n_filters'],
                kernel_size=(fs, dim), padding=(fs//2, 0)) for fs in config['filter_sizes']])
            en_hiddim1 = len(config['filter_sizes']) * config['n_filters'] * 2 + config['ent_embdim']
        else:
            assert(False)

        # sequential container of noise_scorer g(e,m,c)
        self.noise_scorer = nn.Sequential(
                nn.Dropout(p=config['dropout']),
                nn.Linear(en_hiddim1, config['en_dim']),
                nn.ReLU(),
                nn.Dropout(p=config['dropout']),
                nn.Linear(config['en_dim'], 1))
        # sequential container of scorer g(e,m,c)
        self.scorer = nn.Sequential(
                nn.Dropout(p=config['dropout']),
                nn.Linear(en_hiddim1, config['en_dim']),
                nn.ReLU(),
                nn.Dropout(p=config['dropout']),
                nn.Linear(config['en_dim'], 1))

    def forward(self, input):
        N_POSS = input['N_POSS']
        N_NEGS = input['N_NEGS']
        N_CANDS = N_POSS + N_NEGS

        batchsize = input['tokens'].shape[0]

        # position embeddings
        pos_ment_embs = self.pos_embs(input['pos_wrt_m'] + MAX_POS)

        # option to calculate type embedding i.e. term 1/|Te|sum(t), t ∈ Te
        if TYPE_OPT == 'mean':
            # dim: [type_embdim x type_embdim]
            nb_embs = torch.zeros(input['nb_n_types'].shape[0], self.type_embdim).cuda().\
                    scatter_add_(0,
                            input['nb_type_ids'].unsqueeze(1).repeat(1, self.type_embdim),
                            self.type_embs(input['nb_types']))
            nb_embs = nb_embs / input['nb_n_types'].unsqueeze(1).float()

        elif TYPE_OPT == 'max':
            # dim: [ nb_n_types * nb_max_n_types] x type_embdim
            nb_embs = torch.empty(input['nb_n_types'].shape[0] * input['nb_max_n_types'], self.type_embdim).cuda().fill_(-1e10).\
                    scatter_(0,
                            input['nb_type_ids'].unsqueeze(1).repeat(1, self.type_embdim),
                            self.type_embs(input['nb_types']))
            # dim: [ nb_n_types x 1]
            nb_embs = torch.max(nb_embs.view(input['nb_n_types'].shape[0], input['nb_max_n_types'], -1), dim=1)[0]

        rel_embs = self.rel_embs(input['nb_rs'])

        # entity embeddings e = ReLu(We* type-embedding) + be)
        cand_embs = nn.functional.relu(
                torch.zeros(input['cand_n_nb'].shape[0], self.ent_embdim).cuda().\
                        scatter_add_(0,
                            input['cand_nb_ids'].unsqueeze(1).repeat(1, self.ent_embdim),
                            torch.matmul(nb_embs, self.rel_weight) + rel_embs))

        cand_embs = cand_embs.view(batchsize * N_CANDS, -1)
        # batchsize x max_length_list_token_in_batch x dim(vector gloVe)
        inp = self.word_embs(input['tokens'])
        # batchsize x max_length_list_token_in_batch x [dim(vector gloVe)+ pos_embdim]
        inp = torch.cat([inp, pos_ment_embs], dim=2)
        inp = self.dropout(inp)

        if self.type == 'lstm':
            # apply bilstm
            lens = input['masks'].long().sum(dim=1)
            assert(lens[0] == inp.shape[1])

            inp = nn.utils.rnn.pack_padded_sequence(inp, lens, batch_first=True)
            # input inp into BiLSTM
            out, (ht, ct) = self.lstm(inp)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=lens[0])

            out = out.view(batchsize, lens[0], -1).contiguous()
            # context embedding
            ctx_vecs = embedding_3D(out, input['m_loc']).view(batchsize, -1)

        elif self.type == 'pcnn':
            inp = inp.unsqueeze(1)
            conved = [nn.functional.relu(conv(inp)).squeeze(3) for conv in self.convs]  # conved[i]: batchsize x n_filters x len

            # filtering out two parts
            mask = input['pos_wrt_m'].le(0).float().unsqueeze(dim=1)
            left = [c * mask - (1 - mask) * 1e10 for c in conved]
            mask = (input['pos_wrt_m'].ge(0).float() * input['masks']).unsqueeze(dim=1)
            right = [c * mask - (1 - mask) * 1e10 for c in conved]

            # max pooling
            pooled_l = torch.cat([nn.functional.max_pool1d(x, x.shape[2]).squeeze(2) for x in left], dim=1)
            pooled_r = torch.cat([nn.functional.max_pool1d(x, x.shape[2]).squeeze(2) for x in right], dim=1)
            ctx_vecs = torch.cat([pooled_l, pooled_r], dim=1)

        else:
            assert(False)

        rp_ctx_vecs = ctx_vecs.unsqueeze(dim=1).repeat(1, N_CANDS, 1)
        # concatenation between context-mention pair and an entity e
        reprs = torch.cat([rp_ctx_vecs.view(batchsize * N_CANDS, -1), cand_embs], dim=1)
        # score compatibility between (m,c) and an entity e
        scores = self.scorer(reprs).view(batchsize, N_CANDS)
        # masking
        pos_mask = torch.linspace(1, N_POSS, steps=N_POSS).repeat(batchsize, 1).cuda() <= input['real_n_poss'].float().unsqueeze(1).repeat(1, N_POSS)
        mask = torch.cat([pos_mask, torch.ones(batchsize, N_NEGS).cuda().byte()], dim=1)
        scores = torch.where(mask, scores, torch.empty(scores.shape).cuda().fill_(-1e10))

        # compute noise score
        p = torch.nn.functional.softmax(scores[:, :N_POSS])
        # attention weight
        e = (cand_embs.view(batchsize, N_CANDS, -1)[:, :N_POSS, :].contiguous() * p.unsqueeze(dim=2)).sum(dim=1)
        # e(E+)
        reprs = torch.cat([ctx_vecs, e], dim=1)
        noise_scores = self.noise_scorer(reprs)

        return scores, noise_scores