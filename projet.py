import torch
import torch.optim as optim
from vocabulary import Vocabulary
import utils as utils
#from jrk.el import EL
import random
import pickle
import os.path
import numpy as np
import json
import el_hyperparams as hp

args = hp.parser.parse_args()
datadir = args.datadir
#print(args)

#data_path = hp.data_path

# load word embeddins and vocabulary
print('load words and entities')
voca_word, word_embs = utils.load_voca_embs('data/glove/glove.word', 'data/glove/word_embeddings.npy')
word_embs = torch.Tensor(word_embs)
voca_type, _ = Vocabulary.load(datadir + '/agrovoc-type_unique.tsv', normalization=False, add_pad_unk=True)
voca_ent, _ = Vocabulary.load(datadir + '/agrovoc-entity.tsv', normalization=False, add_pad_unk=False)
voca_ent_word, _ = Vocabulary.load(datadir + '/agrovoc-words.lst', normalization=True, add_pad_unk=False, lower=True, digit_0=True)
'''
n_types = voca_type.size()
print(n_types)
type_vocab = vars(voca_type)
print(type_vocab)
'''
'''
n_entity = voca_ent.size()
voca_ent_data = vars(voca_ent_word)
print(voca_ent_data['id2word'])
'''
# load ent2nameId
print('load ent_names')
ent2nameId = {}
f = open(datadir + '/agrovoc-entity.tsv', 'rt')
g = f.read()
h = filter(None,g.split("\n"))
for i in h:
    j = i.split("\t")
    ent2nameId[j[0]] = j[1]

# load ent2typeId
print('load ent2typeId')
ent2typeId = {}
a = open(datadir + '/agrovoc-type-instance.tsv', 'rt')
b = a.read()
c = filter(None,b.split("\n"))
for d in c:
    e = d.split("\t")
    ent2typeId[e[0]] = e[1]

# load triples
print('load triples')
triples_path = datadir + '/agrovoc-triples.tsv'
relId = {}
h2rtId = {}





