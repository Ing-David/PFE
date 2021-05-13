import torch
import torch.nn as nn
import torch.optim as optim

from el import EL
from el_dataset import AGROVOC_DATA
from vocabulary import Vocabulary
import utils as utils
import random
import pickle
import os.path
import numpy as np
import json
import el_hyperparams as hp

# accessing hyper-parameters
args = hp.parser.parse_args()
# directory of data
datadir = args.datadir
# data path
data_path = hp.data_path
# load word embeddings and vocabulary
print('load words and entities')
voca_word, word_embs = utils.load_voca_embs('data/glove/glove.word', 'data/glove/word_embeddings.npy')
# torch.Size([2196019, 300]) 2196019 words with 300 dimensions for each words
word_embs = torch.Tensor(word_embs)
# load types of vocabulary
voca_type, _ = Vocabulary.load(datadir + '/agrovoc-type.tsv', normalization=False, add_pad_unk=True)
# load entities of vocabulary (URI and word)
voca_ent, _ = Vocabulary.load(datadir + '/agrovoc-entity.tsv', normalization=False, add_pad_unk=False)
# load all vocabulary words in Agrovoc
voca_ent_word, _ = Vocabulary.load(datadir + '/agrovoc-words.lst', normalization=True, add_pad_unk=False, lower=True,
                                   digit_0=True)
# number of types of vocabulary in Agrovoc
n_types = voca_type.size()
# load ent2nameId
print('load ent_names')
ent2nameId = {}
with open(datadir +'/agrovoc-entity.tsv', 'rt') as f:
    g = f.read()
    h = filter(None, g.split("\n"))
    for i in h:
        ent2nameId[voca_ent.word2id.get(i)] = i

# load ent2typeId
print('load ent2typeId')
ent2typeId = {}
with open(datadir + '/agrovoc-type-instance.tsv', 'rt') as f:
    g = f.read()
    h = filter(None, g.split("\n"))
    for i in h:
        j = i.split("\t")
        id_entity = voca_ent.word2id.get(j[0])
        id_type = voca_type.get_id(j[1])
        ent2typeId[id_entity] = id_type

# load triples
print('load triples')
triples_path = datadir + '/agrovoc-triples.tsv'
relId = {}
h2rtId = {}

# load dataset
print('load dataset')
dataset = AGROVOC_DATA(data_path,
                       {
                           'word': voca_word,
                           'type': voca_type,
                           'ent': voca_ent
                       },
                       {
                           'ent2typeId': ent2typeId,
                           'ent2nameId': ent2nameId,
                           'relId': relId,
                           'h2rtId': h2rtId,
                       },
                       max_len=args.max_len)

# create model

# training step
if args.mode == 'train':
    print('create model')
    model = EL(config={
        'type': args.enc_type,
        'lstm_hiddim': args.lstm_hiddim,
        'n_filters': args.n_filters,
        'filter_sizes': (3, 5, 7),  # each number has to be odd
        'word_embs': word_embs,
        'pos_embdim': args.pos_embdim,
        'type_embdim': args.type_embdim,
        'ent_embdim': args.ent_embdim,
        'dropout': args.dropout,
        'en_dim': args.en_dim,
        'n_types': n_types,
        'n_rels': len(relId),
        'kl_coef': args.kl_coef,
        'noise_prior': args.noise_prior,
        'margin': args.margin,
    })

# evaluation step
elif args.mode == 'eval':
    print('load model')
    with open(args.model_path + '.config', 'r') as f:
        config = json.load(f)
    print(config)

    model = EL(config={
        'type': config['type'],
        'lstm_hiddim': config['lstm_hiddim'],
        'n_filters': config['n_filters'],
        'filter_sizes': (3, 5, 7),  # each number has to be odd
        'word_embs': word_embs,
        'pos_embdim': config['pos_embdim'],
        'type_embdim': config['type_embdim'],
        'ent_embdim': config['ent_embdim'],
        'dropout': config['dropout'],
        'en_dim': config['en_dim'],
        'n_types': config['n_types'],
        'n_rels': config['n_rels'],
        'kl_coef': config['kl_coef'],
        'noise_prior': config['noise_prior'],
        'margin': config['margin'],
    })
    model.load_state_dict(torch.load(args.model_path + '.state_dict'))

model.cuda()

# for testing
def test(data=None, noise_threshold=args.noise_threshold):
    # check if data is None we take the dev set
    if data is None:
        data = dataset.dev

    # two settings for testing phase

    # all mentions are taken into account
    n_correct_pred = 0
    n_total_pred = 0
    n_total = 0
    # only mentions with Eplus containing the correct entity are considered
    n_correct_pred_or = 0
    n_total_pred_or = 0
    n_total_or = 0

    start = 0

    ner_acc = {}

    while True:
        if start >= len(data):
            break
        # calculate the end value
        end = min(start + args.batchsize, len(data))
        # get the elements from the mini-batch
        input, sents, cands, targets, ners = dataset.get_minibatch(data, start, end)
        # calculate the score and noise_scores
        scores, noise_scores = model(input)
        # probability if a data-point is noisy
        p_noise = torch.nn.functional.sigmoid(noise_scores).cpu().detach().numpy()

        scores = scores.cpu().detach().numpy()

        for pn, ent, sc, cn, ner in zip(p_noise, targets, scores, cands, ners):
            if ner not in ner_acc:
                ner_acc[ner] = {
                    'total': 0,
                    'total_or': 0,
                    'total_pred': 0,
                    'total_pred_or': 0,
                    'correct_pred': 0,
                    'correct_pred_or': 0
                }

            n_total += 1
            ner_acc[ner]['total'] += 1

            in_Eplus = ''
            # cn: list of positive candidates
            if ent in cn:
                n_total_or += 1
                ner_acc[ner]['total_or'] += 1
                in_Eplus = '*'
            # compare p_noise with the hyper-parameter noise_threshold
            if pn > noise_threshold:
                if data == dataset.test:
                    pass  # print('-1' + in_Eplus, end='\t')
                continue
            n_total_pred += 1
            ner_acc[ner]['total_pred'] += 1

            if ent in cn:
                n_total_pred_or += 1
                ner_acc[ner]['total_pred_or'] += 1

            pred = cn[np.argmax(sc)]

            if pred == ent:
                n_correct_pred += 1
                ner_acc[ner]['correct_pred'] += 1
                n_correct_pred_or += 1
                ner_acc[ner]['correct_pred_or'] += 1

                if data == dataset.test:
                    pass  # print('1' + in_Eplus, end='\t')
            else:
                if data == dataset.test:
                    pass  # print('0' + in_Eplus, end='\t')

        if data == dataset.test:
            pass  # print()
        # take another batch
        start = end

    # precision, recall, and f1-score (All mentions are taken into account
    prec = n_correct_pred / n_total_pred
    rec = n_correct_pred / n_total
    try:
        f1 = 2 * (prec * rec) / (prec + rec)
    except:
        f1 = 0
    print('all -- prec: %.2f\trec: %.2f\tf1: %.2f' % (prec * 100, rec * 100, f1 * 100))

    # precision, recall, and f1-score (Only mentions with E+ containing the correct entity are considered)
    prec = n_correct_pred_or / n_total_pred_or
    rec = n_correct_pred_or / n_total_or
    try:
        f1 = 2 * (prec * rec) / (prec + rec)
    except:
        f1 = 0
    print('in E+', n_total_or / n_total * 100)
    print('in E+ -- prec: %.2f\trec: %.2f\tf1: %.2f' % (prec * 100, rec * 100, f1 * 100))

    print('ner')
    print(ner_acc)
    return prec, rec, f1


# for training
def train():
    # parameters of the model
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer Adam
    optimizer = optim.Adam(params, lr=args.lr)
    data = dataset.train

    best_scores = {'prec': -1, 'rec': -1, 'f1': -1}

    if args.kl_coef > 0:
        print('*** dev ***')
        test(dataset.dev)
        print('*** test ***')
        test(dataset.test)

    print('===== noise_threshold=1 ====')
    print('*** dev ***')
    test(dataset.dev, noise_threshold=1)
    print('*** test ***')
    test(dataset.test, noise_threshold=1)

    for e in range(args.n_epochs):
        print('------------------------- epoch %d --------------------------' % (e))
        random.shuffle(data)
        model.train()
        start = end = 0
        total_loss = 0

        while True:
            if start >= len(data):
                print('%.6f\t\t\t\t\t' % (total_loss / len(data)))
                save = False
                if args.kl_coef > 0:
                    print('*** dev ***')
                    prec, rec, f1 = test(dataset.dev)
                    print('*** test ***')
                    test(dataset.test)

                    if best_scores['f1'] <= f1:
                        best_scores = {'prec': prec, 'rec': rec, 'f1': f1}
                        save = True

                print('===== noise_threshold=0 ====')
                print('*** dev ***')
                prec, rec, f1 = test(dataset.dev, noise_threshold=1)
                print('*** test ***')
                test(dataset.test, noise_threshold=1)

                if args.kl_coef == 0 and best_scores['f1'] <= f1:
                    best_scores = {'prec': prec, 'rec': rec, 'f1': f1}
                    save = True

                if save:
                    print('save model to', args.model_path)
                    model.save(args.model_path)

                break

            end = min(start + args.batchsize, len(data))
            input, sents, cands, _, _ = dataset.get_minibatch(data, start, end)

            optimizer.zero_grad()
            # score and noise_scores from the model
            scores, noise_scores = model(input)
            # compute the loss
            loss, kl = model.compute_loss({
                'scores': scores,
                'noise_scores': noise_scores,
                'real_n_poss': input['real_n_poss'],
                'N_POSS': input['N_POSS']})
            # computer gradient of loss wrt all parameters in loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5)  # 5 arbitrary value of choosing
            # update values of all parameters in loss to get the minimum loss
            optimizer.step()
            loss = loss.data.cpu().item()

            # print sentence
            if False and end < 1001:
                p_noise = torch.nn.functional.sigmoid(noise_scores)
                for _i in range(end - start):
                    p_noise_i = p_noise[_i]
                    if True:  # p_noise_i > args.noise_threshold:
                        scores_i = scores[_i][:input['N_POSS']]
                        sent_i = sents[_i]
                        m_loc_i = input['m_loc'][_i]
                        cands_i = cands[_i][:input['N_POSS']]
                        n_poss_i = input['real_n_poss'][_i].item()

                        words = sent_i.split(' ')
                        words[m_loc_i[0]] = '[' + words[m_loc_i[0]]
                        words[m_loc_i[1] - 1] = words[m_loc_i[1] - 1] + ']'
                        sent_i = ' '.join(words)

                        best_score, best_pred = torch.max(scores_i, dim=0)
                        best_score = best_score.cpu().item()
                        best_pred = best_pred.cpu().item()
                        best_entId = cands_i[best_pred]
                        best_ent = voca_ent.id2word[best_entId]
                        best_types = [voca_type.id2word[t] for t in ent2typeId[best_entId]]
                        best_name = [voca_ent_word.id2word[w] for w in ent2nameId[best_entId]]

                        print('------------------ data point ---------------')
                        print(p_noise_i)
                        print(sent_i)
                        print(n_poss_i)
                        print(best_ent, best_name, best_types, best_score)

                        print('CANDS')
                        for _j in range(n_poss_i):
                            entId_ij = cands_i[_j]
                            ent_ij = voca_ent.id2word[entId_ij]
                            types_ij = [voca_type.id2word[t] for t in ent2typeId[entId_ij]]
                            name_ij = [voca_ent_word.id2word[w] for w in ent2nameId[entId_ij]]
                            print('\t', ent_ij, name_ij, types_ij)

            print("%d\tloss=%.6f\tkl=%.6f\t\t\t" % (end, loss, kl), end='\r' if random.random() < 0.995 else '\n')

            total_loss += loss * (end - start)
            start = end


if __name__ == '__main__':
    if args.mode == 'train':
        train()

    elif args.mode == 'eval':
        if model.config['kl_coef'] > 0:
            print('*** dev ***')
            test(dataset.dev)
            print('*** test ***')
            test(dataset.test)

        print('===== noise_threshold=0 ====')
        print('*** dev ***')
        test(dataset.dev, noise_threshold=1)
        print('*** test ***')
        test(dataset.test, noise_threshold=1)

    else:
        assert (False)
