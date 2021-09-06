import el_hyperparams as hp
import utils as utils
import torch
from torch.autograd import Variable
import json
import random
from copy import deepcopy

class ELDataset:

    def __init__(self, data_path, vocas, triples, max_len=100):
        self.vocas = vocas
        self.triples = triples
        self.entIdList = list(self.triples['ent2typeId'].keys())

        if 'self' not in self.triples['relId']:
            self.triples['relId']['self'] = len(self.triples['relId'])

        print('load train set')
        self.train = self.read_from_file(data_path['train'], max_len=max_len)
#        print('load dev set')
#        self.dev = self.read_from_file(data_path['dev'], train=False, max_len=max_len)
#        print('load test set')
#        self.test = self.read_from_file(data_path['test'], train=False, max_len=max_len)

    def read_from_file(self, path, format='json', train=True, max_len=100):
        pass

    def get_minibatch(self, data, start, end):

        # verify if data is a training set
        if data == self.train:
            MAX_N_POSS = hp.MAX_N_POSS_TRAIN
        else:
            MAX_N_POSS = hp.MAX_N_POSS_TEST

        # take the data from index start to index end consider as a batch
        org = data[start:end]
        # sort by number of words in descending order
        org.sort(key=lambda x: len(x[0]), reverse=True)
        # dictionary input
        input = {
                'tokens': [], 'masks': [], 'm_loc': [], 'pos_wrt_m': [],
                'nb_types': [], 'nb_type_ids': [], 'nb_n_types': [],
                'nb_rs': [], 'cand_n_nb': [], 'cand_nb_ids': [],
                'real_n_poss': [],
                }
        # list sentence
        sentence = []
        # list candidate
        candidates = []
        # find the minimum length for each list of positive candidates compare with NAX_N_POSS in a batch and put
        # them in a list
        input['real_n_poss'] = [min(len(x[4]), MAX_N_POSS) for x in org]
        # find maximum number of positive candidates in a batch
        input['N_POSS'] = max(input['real_n_poss'])
        # list targeted ent
        targets = []
        # list ners
        ners = []
        # find the minimum length for each list of negative candidate in a batch and put them in a list
        input['real_n_negs'] = [len(x[5]) for x in org]
        # find maximum number of positive candidates in a batch
        input['N_NEGS'] = max(input['real_n_negs'])
        # if the negative candidates don't exist then take the value of hyper-parameter
        if input['N_NEGS'] == 0:
            input['N_NEGS'] = hp.N_NEGS
    # ITEM is the tuple of (id_word,(mention_start,mention_end),pos_wrt_m,sentence,pos_can,neg_can, entity_id,ner_id)
        # loop each item in the batch org
        for item in org:
            # copy all values from item into variables (tokens, m_loc...)
            tokens, m_loc, pos_wrt_m, sent, positives, negatives, ent, ner = deepcopy(item)
            
	    # condition for sampling the negative candidate
            if hp.SAMPLE_NEGS:
                negatives = random.sample(self.entIdList, hp.N_NEGS)
            else:
                if data == self.train:
                    if len(negatives) == 0:
                        negatives = random.sample(self.entIdList, input['N_NEGS'])
                    else:
                        negatives = negatives + [negatives[-1]] * (input['N_NEGS'] - len(negatives))
                else:
                    negatives = negatives

            # get ids of parents (skos:broader) of each candidate in the positive and negative list for each item
            neg_types = [self.triples['ent2typeId'][c] for c in negatives]
            pos_types = [self.triples['ent2typeId'][c] for c in positives]
	    
		
            # verify if the length of positive list exceed the define length
            if len(positives) > input['N_POSS']:
                # adjust the number of candidate in list equal to the size of define length
                positives = positives[:input['N_POSS']]
                pos_types = pos_types[:input['N_POSS']]
            else:
                # add the last element in the list multiple times to match the N_POSS
                positives += [positives[-1]] * (input['N_POSS'] - len(positives))
                pos_types += [pos_types[-1]] * (input['N_POSS'] - len(pos_types))

            # total types of candidate(pos and neg) for each item
            cand_types = pos_types + neg_types
            # list of token's ids for each item
            input['tokens'].append(tokens)
            # list of tuple of mentions for each item
            input['m_loc'].append(m_loc)
            # list of pos_wrt_m for each item
            input['pos_wrt_m'].append(pos_wrt_m)
            # list contains all types of candidates(pos_can and neg_can) for each item
            input['nb_types'].extend([[types] for types in cand_types])
            # list with the length in function of number of relations in the RDF triple that the types of candidates(
            # pos_can and neg_can) have with other entities for each item
            input['nb_rs'].extend([[self.triples['relId']['self']] for t in cand_types])
            # list ids of all candidates(pos_can and neg_can) for each item
            candidates.append(positives + negatives)
            # list that contains sentence of each item
            sentence.append(sent)
            # list targeted ent of each item
            targets.append(ent)
            # list ners of each item
            ners.append(ner)

        # summary data in a batch
        # get neighbour
        # nested generator to convert list of many lists to one list of candidates for all items in a batch
        flat_candidates = [c for cands in candidates for c in cands]
        for c, nb_types, nb_rs in zip(flat_candidates, input['nb_types'], input['nb_rs']):
            # this condition matches if there is a dictionary h2rtId to verify with the variable c
            if c in self.triples['h2rtId'] and len(self.triples['h2rtId'][c]) < 30:
                tmp = [(rt >> 32, rt - ((rt >> 32) << 32)) for rt in self.triples['h2rtId'][c]]
                nb_types += [self.triples['ent2typeId'][t] for _, t in tmp if t in self.triples['ent2typeId']]
                nb_rs += [r for r, t in tmp if t in self.triples['ent2typeId']]
            # list of number of types for candidates in a batch
            input['cand_n_nb'].append(len(nb_types))
            # list contains the iterative lists(0,1,..,end) with length in function of number of relation candidates
            # with other entities in a batch
            input['cand_nb_ids'].append([len(input['cand_nb_ids'])] * len(nb_rs))
        # flatten to one list of nb_types of candidates in a batch
        #nput['nb_types'] = [types for nb_t in input['nb_types'] for types in nb_t]
        # list contains the length of types that candidates(all items) have in a batch
        input['nb_n_types'] = [len(types) for types in input['nb_types']]

	# flatten to one list of nb_types of candidates in a batch
        #input['nb_types'] = [types for nb_t in input['nb_types'] for types in nb_t]

        # option to represent the list ids of number of types of candidates for all items in a batch
        if hp.TYPE_OPT == 'mean':
            input['nb_type_ids'] = [[i] * len(types) for i, types in enumerate(input['nb_types'])]
        elif hp.TYPE_OPT == 'max':
            input['nb_max_n_types'] = max(input['nb_n_types'])
            input['nb_type_ids'] = [list(range(i * input['nb_max_n_types'], i * input['nb_max_n_types'] + len(types)))
                    for i, types in enumerate(input['nb_types'])]

        # flatten using nested generator to convert list of list to one list of all items
        input['nb_types'] = [t for types in input['nb_types'] for t in types]
        input['nb_type_ids'] = [_i for _ids in input['nb_type_ids'] for _i in _ids]
        input['nb_rs'] = [r for rs in input['nb_rs'] for r in rs]
        input['cand_nb_ids'] = [_i for _ids in input['cand_nb_ids'] for _i in _ids]

        # convert to Tensor 64 bit integer(CPU tensor)

        # make equal size(max_len) for all the lists of tokens in a batch
        input['tokens'], input['masks'] = utils.make_equal_len(input['tokens'], fill_in=self.vocas['word'].pad_id)
        # dim: batch-size x len(max_len of all lists in a batch)
        input['tokens'] = Variable(torch.LongTensor(input['tokens']).cuda(), requires_grad=False)
        input['masks'] = Variable(torch.Tensor(input['masks']).cuda(), requires_grad=False)
        # dim: batch-size x len(each mentioned tuple (ms,me) i.e 2 )
        input['m_loc'] = Variable(torch.LongTensor(input['m_loc']).cuda(), requires_grad=False)
        # make equal size(max_len) for all the lists of pos_wrt_m in a batch
        input['pos_wrt_m'], _ = utils.make_equal_len(input['pos_wrt_m'], fill_in=hp.MAX_POS)
        # dim: batchsize x len(max_len of all lists pos_wrt_m)
        input['pos_wrt_m'] = Variable(torch.LongTensor(input['pos_wrt_m']).cuda(), requires_grad=False)
        # dim: batch-size x 1
        input['nb_types'] = Variable(torch.LongTensor(input['nb_types']).cuda(), requires_grad=False)
        # dim: batch-size
        input['nb_n_types'] = Variable(torch.LongTensor(input['nb_n_types']).cuda(), requires_grad=False)
        # dim: batch-size
        input['nb_type_ids'] = Variable(torch.LongTensor(input['nb_type_ids']).cuda(), requires_grad=False)
        # dim: batch-size
        input['cand_n_nb'] = Variable(torch.LongTensor(input['cand_n_nb']).cuda(), requires_grad=False)
        # dim: [batch-size x batch-size]
        input['cand_nb_ids'] = Variable(torch.LongTensor(input['cand_nb_ids']).cuda(), requires_grad=False)
        # dim: batch-size x 1
        input['nb_rs'] = Variable(torch.LongTensor(input['nb_rs']).cuda(), requires_grad=False)
        # dim: batch-size
        input['real_n_poss'] = Variable(torch.LongTensor(input['real_n_poss']).cuda(), requires_grad=False)

        return input, sentence, candidates, targets, ners


class AGROVOC_DATA(ELDataset):
    """
    Inherits from class ELDataset
    """
    def __init__(self, data_path, vocas, triples, max_len=100):
        data_path = {
                'train': data_path,
                'dev': 'data/EL/',
                'test': 'data/EL/'
                }
        super(AGROVOC_DATA, self).__init__(data_path, vocas, triples, max_len=max_len)

    def read_json(self, path):
        """
        function to read raw data from file json
        """
        data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i % int(1e3) == 0:
                    print(i, end='\r')
                line = line.strip()
                data.append(json.loads(line))

        return data

    def read_data(self, path, format='json', train=True, max_len=100):

        if train:
            MAX_N_POSS = hp.MAX_N_POSS_TRAIN
        else:
            MAX_N_POSS = hp.MAX_N_POSS_TEST

        # list data
        data = []
        print('read file from', path)
        # verify if the file is a json file
        if format == 'json':
            # read json file
            raw_data = self.read_json(path)
        else:
            assert(False)
        # mode train
        if train:
            # reorganize the order of the list in raw_data
            random.shuffle(raw_data)

        print('load data')
        for count, item in enumerate(raw_data):
            if format == 'json':
                # access to list of key sentence and mentions respectively
                org_words, ments = item['sentence'], item['mentions']
                # join token to get a whole sentence
                sent = ' '.join(org_words)
                # get the id of each word in org_words
                words = [self.vocas['word'].get_id(w) for w in org_words]

                # skip sentence that has the length longer than the max_len
                if len(words) > max_len:
                    continue
                for ment in ments:
                    # access mention start and mention end value
                    ms, me = ment['mention']
                    # no id found in a sentence
                    if len(words) == 0:
                        print(sent)
                        continue
                    # find position with respect to mention of each mention in a sentence
                    pos_wrt_m = [max(i - ms, -hp.MAX_POS) for i in range(0, ms)] + \
                            [0] * (me - ms) + \
                            [min(i - me + 1, hp.MAX_POS) for i in range(me, len(words))]

                    # list of positive candidate found in dict ent2typeId
                    positives = [c for c in ment['positives'] if c in self.triples['ent2typeId']]
                    # list of negative candidate found in dict ent2typeId
                    negatives = [c for c in ment['negatives'] if c in self.triples['ent2typeId']]

                    # list positive is empty then skip
                    if len(positives) == 0:
                        continue
                    # append tuple of (id_word,(mention_start,mention_end),pos_wrt_m,sentence,pos_can,neg_can,
                    # entity_id,ner_id)
                    data.append((words, (ms, me), pos_wrt_m, sent, positives, negatives, ment.get('entity', None), ment.get('ner', 'O')))

                if (count + 1) % 1000 == 0:
                    print(count // 1000, 'k', end='\r')

        print('load', len(data), 'items')
        return data

    def read_from_file(self, path, format='json', train=True, max_len=100):
        return self.read_data(path, format=format, train=train, max_len=max_len)
