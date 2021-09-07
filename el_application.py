import numpy as np
import json
import re
import tempfile
import el_hyperparams as hp
import utils as utils
import torch
from annotation import Agrovoc
from vocabulary import Vocabulary
from tqdm.notebook import tqdm
tqdm.pandas()
from nltk.tokenize import word_tokenize
from operator import itemgetter
from copy import deepcopy
from torch.autograd import Variable
from el import EL

import logging
logger = logging.getLogger(__name__)


def contains_word(s, w):
    """
    function to check wheather a word contain in a string
    :param s: string to check
    :param w: word to verify
    """
    s = re.sub('[):,.?!(]', '', s)
    s = re.sub('\-', ' ', s)
    return (' ' + w + ' ') in (' ' + s + ' ')

def read_json(path):
    """
    function to read raw data from file json
    :param path: path to datapoint
    """
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i % int(1e3) == 0:
                print(i, end='\r')
            line = line.strip()
            data.append(json.loads(line))

    return data

def read_data(path, voca_word, ent2typeId,  format='json',  max_len=512):
    """
        function to read raw data from file json and convert it to datapoint before entering in get_minibatch
        :param path: path to datapoint
        :param voca_word: vocabulary from GLoVe
        :param format: of the file
        :param max_len: maximum length of each sentence
    """
    # list data
    data = []
    # verify if the file is a json file
    if format == 'json':
        # read json file
        raw_data = read_json(path)
    else:
        assert (False)
    # print('load data')
    for count, item in enumerate(raw_data):
        if format == 'json':
            # access to list of key sentence and mentions respectively
            org_words, ments = item['sentence'], item['mentions']
            # join token to get a whole sentence
            sent = ' '.join(org_words)
            # get the id of each word in org_words
            words = [voca_word.get_id(w) for w in org_words]

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
                positives = [c for c in ment['positives'] if c in ent2typeId]
                # list of negative candidate found in dict ent2typeId
                negatives = [c for c in ment['negatives'] if c in ent2typeId]

                # list positive is empty then skip
                if len(positives) == 0:
                    continue
                # append tuple of (id_word,(mention_start,mention_end),pos_wrt_m,sentence,pos_can,neg_can,
                # entity_id,ner_id)
                data.append((words, (ms, me), pos_wrt_m, sent, positives, negatives, ment.get('entity', None),
                             ment.get('ner', 'O')))

            if (count + 1) % 1000 == 0:
                print(count // 1000, 'k', end='\r')

    return data

def get_minibatch(data, ent2typeId, voca_word, relId, h2rtId,  start, end):
    """
        function to read  datapoint from function read_data
        :param ent2typeId: load ent2typeId
        :param voca_word: vocabulary from GLoVe
        :param start: start of datapoint
        :param end: end of datapoint
    """

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
    input['real_n_poss'] = [min(len(x[4]), hp.MAX_N_POSS_TEST) for x in org]
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
    
    # ITEM is the tuple of (id_word,(mention_start,mention_end),pos_wrt_m,sentence,pos_can,neg_can, entity_id,ner_id)
    # loop each item in the batch org
    for item in org:
        # copy all values from item into variables (tokens, m_loc...)
        tokens, m_loc, pos_wrt_m, sent, positives, negatives, ent, ner = deepcopy(item)
        # get ids of parents (skos:broader) of each candidate in the positive and negative list for each item
        neg_types = [ent2typeId[c] for c in negatives]
        pos_types = [ent2typeId[c] for c in positives]
        # condition for sampling the positive candidates
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
        input['nb_rs'].extend([[relId['self']] for t in cand_types])
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
        if c in h2rtId and len(h2rtId[c]) < 30:
            tmp = [(rt >> 32, rt - ((rt >> 32) << 32)) for rt in h2rtId[c]]
            nb_types += [ent2typeId[t] for _, t in tmp if t in ent2typeId]
            nb_rs += [r for r, t in tmp if t in ent2typeId]
        # list of number of types for candidates in a batch
        input['cand_n_nb'].append(len(nb_types))
        # list contains the iterative lists(0,1,..,end) with length in function of number of relation candidates
        # with other entities in a batch
        input['cand_nb_ids'].append([len(input['cand_nb_ids'])] * len(nb_rs))
    # list contains the length of types that candidates(all items) have in a batch
    input['nb_n_types'] = [len(types) for types in input['nb_types']]

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
    input['tokens'], input['masks'] = utils.make_equal_len(input['tokens'], fill_in=voca_word.pad_id)
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


def application_entity_linking(text, batchsize = 5):
    """
        function to annotated text with dictionary agrovoc
        :param text: text to annotate
        :param batchsize: batchsize to process for mentions that have more than one URI to get a unique URI define by the model
    """

    # accessing hyper-parameters
    args = hp.parser.parse_args()
    # directory of data
    datadir = 'data/agrovoc'

    # Thesaurus AGROVOC
    agrovoc = Agrovoc(lang="en")
    # load entities of vocabulary (URI and word)
    voca_ent, _ = Vocabulary.load(datadir + '/agrovoc-entity.tsv', normalization=False, add_pad_unk=False)
    # load types of vocabulary
    voca_type, _ = Vocabulary.load(datadir + '/agrovoc-type.tsv', normalization=False, add_pad_unk=True)

    logger.info('load words and entities')
    voca_word, word_embs = utils.load_voca_embs('data/glove/glove.word', 'data/glove/word_embeddings.npy')
    word_embs = torch.Tensor(word_embs)

    # load triples
    logger.info('load triples')
    triples_path = datadir + '/agrovoc-triples.tsv'
    relId = {}
    h2rtId = {}

    if 'self' not in relId:
        relId['self'] = len(relId)

    # load ent2nameId
    logger.info('load ent_names')
    ent2nameId = {}
    with open(datadir + '/agrovoc-entity.tsv', 'rt') as f:
        g = f.read()
        h = filter(None, g.split("\n"))
        for i in h:
            ent2nameId[voca_ent.word2id.get(i)] = i

    # load ent2typeId
    logger.info('load ent2typeId')
    ent2typeId = {}
    with open(datadir + '/agrovoc-type-instance.tsv', 'rt') as f:
        g = f.read()
        h = filter(None, g.split("\n"))
        for i in h:
            j = i.split("\t")
            id_entity = voca_ent.word2id.get(j[0])
            id_type = voca_type.get_id(j[1])
            ent2typeId[id_entity] = id_type



    # annotated text by annotation by dictionary
    annotated_text = agrovoc.annotate_text(text)
    output = {}
    output['text'] = text
    output['entities'] = []

    first_list_phrase = re.split(r'(?<=\.)\s+(?=[a-zA-Z])', text)
    new_string = " ".join(first_list_phrase)
    list_phrase = re.split(r'(?<=\.)\s+(?=[a-zA-Z])', new_string)
    list_length_phrase = []

    # get the length of all phrases
    for a in list_phrase:
        list_length_phrase.append(len(a))

    offset_phrase = []

    # loop to get the offset of all phrases in the text (assume that each phrase separated by ' ')
    for index, length_phrase in enumerate(list_length_phrase):
        # first element in the list
        if index == 0:
            offset_phrase.append((0, length_phrase))
        # second element and so on...
        else:
            offset_phrase.append((sum(i for i in list_length_phrase[0:index]) + index,
                                  sum(i for i in list_length_phrase[0:index + 1]) + index))

    # get some properties from tool annotation by dictionary
    for mention in annotated_text[2]:
        dict_entity = {}
        dict_mention = vars(mention)
        dict_entity["entity"] = dict_mention["matched_text"]
        dict_entity["offsetStart"] = dict_mention["start"]
        dict_entity["offsetEnd"] = dict_mention["end"]
        dict_entity["agrovoc_uri"] = dict_mention["concept_id"]
        output['entities'].append(dict_entity)

    # combine a mention with more than one URIs
    tmp = {}
    for d in output['entities']:
        fields = d['entity'], d['offsetStart'], d['offsetEnd']
        id_ = d['agrovoc_uri']
        if fields in tmp:
            tmp[fields]['agrovoc_uri'].append(id_)
        else:
            d_copy = d.copy()
            d_copy['agrovoc_uri'] = [id_]
            tmp[fields] = d_copy

    output['entities'] = list(tmp.values())

    # filter to keep only mention that contains more than one URI in order to be decided by the model
    enter_model = [d for d in output['entities'] if len(d['agrovoc_uri']) > 1]

    # filter to keep only unique URI for each mention
    output['entities'][:] = [d for d in output['entities'] if len(d['agrovoc_uri']) == 1]

    # loop to get the original sentence of each mention
    for dict_entity in enter_model:
        for index, phrase in enumerate(list_phrase):
            if offset_phrase[index][0] <= dict_entity['offsetStart'] and dict_entity['offsetEnd'] <= \
                    offset_phrase[index][1]:
                dict_entity['original_sentence'] = list_phrase[index]

    list_sentence_json = []

    # loop to get the json line for each mentioned datapoint
    for dict_concept in enter_model:
        concept_dict = {}
        dict_mention = {}
        # convert string to list of tokens
        concept_dict["sentence"] = word_tokenize(dict_concept['original_sentence'])
        concept_dict['mentions'] = []
        # get the position in the list of mentioned concept
        for token in concept_dict['sentence']:
            if contains_word(token.lower(), dict_concept['entity'].lower()):
                dict_mention['mention'] = [concept_dict["sentence"].index(token),
                                           concept_dict["sentence"].index(token) + 1]

        list_concept_position = []
        for uri in dict_concept['agrovoc_uri']:
            concept_position = voca_ent.word2id.get(uri)
            list_concept_position.append(concept_position)

        dict_mention['positives'] = list_concept_position
        dict_mention['negatives'] = []
        concept_dict['mentions'].append(dict_mention)
        list_sentence_json.append(concept_dict)

    # write json line into temporary file
    tmp = tempfile.NamedTemporaryFile()

    data = []
    for line_json in list_sentence_json:
        with open(tmp.name, 'w') as f:
            json.dump(line_json, f)
            f.write('\n')

        with open(tmp.name) as f:
            datapoint = read_data(f.name, voca_word, ent2typeId)
            data.append(datapoint[0])

    logger.info("Total data point to process in the model: ", len(data))

    logger.info('load model')
    with open(args.model_path + '/config', 'r') as f:
        config = json.load(f)

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
    model.load_state_dict(torch.load(args.model_path + '/state_dict'))

    model.cuda()

    start = 0

    # list to store URI decided by the model
    decided_entity = []

    while True:
        if start >= len(data):
            break
        # calculate the end value
        end = min(start + batchsize, len(data))
        # get the elements from the mini-batch
        input, sents, cands, targets, ners = get_minibatch(data, ent2typeId, voca_word, relId, h2rtId, start, end)
        # calculate the score and noise_scores
        scores, noise_scores = model(input)
        # probability if a data-point is noisy
        p_noise = torch.nn.functional.sigmoid(noise_scores).cpu().detach().numpy()

        scores = scores.cpu().detach().numpy()

        for pn, ent, sc, cn, ner in zip(p_noise, targets, scores, cands, ners):

            potential_entity = cn[np.argmax(sc)]
            decided_entity.append(voca_ent.id2word[potential_entity])

    # Loop to get the entity decided by the model
    for entered_model, gold_entity in zip(enter_model, decided_entity):
        entered_model['agrovoc_uri'] = [gold_entity]

    # delete key original sentence
    for dict_entity in enter_model:
        del dict_entity['original_sentence']

    # conbine list of unique URI and list of unique URI decided by the model
    output['entities'] = output['entities'] + enter_model

    # sort the dictionary based on offsetStart
    output['entities'] = sorted(output['entities'], key=itemgetter('offsetStart'))

    # convert list of one string to string
    for dict_concept in output['entities']:
        dict_concept['agrovoc_uri'] = dict_concept['agrovoc_uri'][0]

    return output

#text_to_annotate = "...."
#output = application_entity_linking(text=text_to_annotate,batchsize = 5)
