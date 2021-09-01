import json
from annotation import Agrovoc
from vocabulary import Vocabulary
from itertools import chain
from collections import defaultdict
from operator import itemgetter
import re
from tqdm.notebook import tqdm

tqdm.pandas()

# directory of data
datadir = 'data/agrovoc'


def individual_tokenOffset(text, pattern):
    """
    Function to get the offset of a mention in an individual sentence, useful with the offset of tool annotation by dictionary
    :param text: The input sentence
    :param pattern: The list of pattern (i.e a word/mention we want to find its offset in the sentence)
    """

    items = []
    for m in pattern.finditer(text):
        item = {}
        item['word'] = m.group()
        item['IndividualOffsetBegin'] = m.start()
        item['IndividualOffsetEnd'] = m.end()
        items.append(item)
    return items


def contains_word(s, w):
    s = re.sub('[):,.?!(]', '', s)
    return (' ' + w + ' ') in (' ' + s + ' ')

def list_dict_mention(text_file, brat_file):
    """
    Function to generate the list of dictionary contains each mention's properties
    :param text_file: the file with the extension (.txt)
    :param brat_file: the output file from BRAT's tool with extension (.ann)

    Example of an output list::

      [{'standoff_id': 1,
      'entity_type': 'Concept',
      'offset_start': 13,
      'offset_end': 18,
      'word': 'wheat',
      'concept_id': 'http://aims.fao.org/aos/agrovoc/c_8373',
      'original_sentence': 'Ceci est une wheat phrase corn.\n',
      'start': 13,
      'end': 18},
     {'standoff_id': 2,
      'entity_type': 'Concept',
      'offset_start': 26,
      'offset_end': 30,
      'word': 'corn',
      'concept_id': 'http://aims.fao.org/aos/agrovoc/c_12332',
      'original_sentence': 'Ceci est une wheat phrase corn.\n',
      'start': 26,
      'end': 30},
     {'standoff_id': 3,
      'entity_type': 'Concept',
      'offset_start': 61,
      'offset_end': 67,
      'word': 'barley',
      'concept_id': 'http://aims.fao.org/aos/agrovoc/c_823',
      'original_sentence': 'Ceci est une deuxième phrase barley.',
      'start': 29,
      'end': 35}]
    """
    # looping through .txt files
    with open(text_file) as f:

        first_list_phrase = re.split(r'(?<=\.)\s+(?=[a-zA-Z])', f.read())
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
            offset_phrase.append(
                (sum(i for i in list_length_phrase[0:index]) + index,
                 sum(i for i in list_length_phrase[0:index + 1]) + index))

    # looping through .ann files in the data directory
    STANDOFF_ENTITY_PREFIX = 'T'
    STANDOFF_RELATION_PREFIX = 'N'
    entities = []
    relations = []
    with open(brat_file, 'r') as document_anno_file:
        lines = document_anno_file.readlines()
        for line in lines:
            standoff_line = line.split()
            if standoff_line[0][0] == STANDOFF_ENTITY_PREFIX:
                entity = {}
                entity['standoff_id'] = int(standoff_line[0][1:])
                entity['entity_type'] = standoff_line[1].capitalize()
                entity['offset_start'] = int(standoff_line[2])
                entity['offset_end'] = int(standoff_line[3])
                list_word = []
                for i in range(4, len(standoff_line)):
                    list_word.append(standoff_line[i])
                entity['word'] = " ".join(list_word)
                entities.append(entity)

            elif standoff_line[0][0] == STANDOFF_RELATION_PREFIX:
                relation = {}
                relation['standoff_id'] = int(standoff_line[0][1:])
                relation['concept_id'] = standoff_line[3].replace("Agrovoc:", "")
                relations.append(relation)

    # loop to merge dictionary of entities and relations
    d = defaultdict(dict)
    for l in (entities, relations):
        for elem in l:
            d[elem["standoff_id"]].update(elem)

    list_dict_concept = sorted(d.values(), key=itemgetter("standoff_id"))

    # loop to get the original sentence for each mention
    for dict_concept in list_dict_concept:
        for index, phrase in enumerate(list_phrase):

            if contains_word(phrase, dict_concept['word']) and offset_phrase[index][0] <= dict_concept[
                'offset_start'] and dict_concept['offset_end'] <= offset_phrase[index][1]:
                dict_concept['original_sentence'] = list_phrase[index]

    # In case some mention cannot find its original sentence
    keys = set(chain.from_iterable(list_dict_concept))
    for item in list_dict_concept:
        item.update({key: "" for key in keys if key not in item})

        # Remove all dictionary whose mention cannot find its original sentence or sentence id
    list_dict_concept[:] = [d for d in list_dict_concept if d.get('original_sentence') != ""]

    # loop to get the offset of each mention for individual sentence
    for con_dict in list_dict_concept:
        text = con_dict["original_sentence"]
        tokens = [con_dict['word']]
        pattern = re.compile(
            fr'(?<!\w)(?:{"|".join(sorted(map(re.escape, tokens), key=len, reverse=True))})(?!\w)(?!\.\b)', re.I)
        offsets = individual_tokenOffset(text, pattern)
        list_start_offset = []
        list_end_offset = []

        for offset in offsets:
            list_start_offset.append(offset['IndividualOffsetBegin'])
            list_end_offset.append(offset['IndividualOffsetEnd'])

        con_dict["start"] = list_start_offset
        con_dict["end"] = list_end_offset

    return list_dict_concept


def json_test_file(text_file, brat_file, output_json):
    """
    Function to generate the json file for testing the model dl4el
    :param text_file: The text file (.txt)
    :param brat_file: The output file from the BRAT's tool (.ann) :param output_json: The name of the output json file,
    the name should not be changed if we want a file with continuous line
    """
    # Agrovoc's rdf
    agrovoc = Agrovoc(lang="en")
    # access dictionary of agrovoc with correspond id for searching the index location of id for candidate positive and gold label(entity)
    voca_ent, _ = Vocabulary.load(datadir + '/agrovoc-entity.tsv', normalization=False, add_pad_unk=False)
    # get the entIdList for random negative candidates
    ent2nameId = {}
    with open(datadir + '/agrovoc-entity.tsv', 'rt') as f:
        g = f.read()
        h = filter(None, g.split("\n"))
        for i in h:
            ent2nameId[voca_ent.word2id.get(i)] = i

    list_dict_concept = list_dict_mention(text_file, brat_file)

    list_line_json = []

    # loop through each sentence from list_dict_concept
    for index, sentence in enumerate(list_dict_concept):

        dict_phrase = {}
        string_sentence = sentence['original_sentence'].lower()
        # annotated concepts for each phrase
        output = agrovoc.annotate_text(string_sentence)
        # all tokens of each phrase
        dict_phrase["sentence"] = output[1]
        # list mention for each phrase
        dict_phrase["mentions"] = []
        # The offsets for each token
        token_offset = output[0]
        # list of annotated concepts
        annotated_concepts = []
        for annot_concept in output[2]:
            value = vars(annot_concept)
            annotated_concepts.append(value)
        # access position of each annotations
        list_pos_tok = []  # list position of each annotated concept in each phrase
        list_id_tok = []  # list URI of each annotated concept in each phrase
        for concept in annotated_concepts:
            # get position of annotated concept
            list_pos_tok.append((concept["start"], concept["end"]))
            # get ids of annotated concept
            list_id_tok.append((concept["concept_id"], concept["matched_text"]))
        # remove duplicated tuple from the list
        list_id_tok = [t for t in (set(tuple(i) for i in list_id_tok))]
        # remove the same mention index from the list
        list_pos_tok = list(dict.fromkeys(list_pos_tok))

        # loop through each mention in a phrase
        for pos_tok in list_pos_tok:
            # dictionary mention
            dict_mention = {}
            # position of each mentioned token in list of all tokens [start_index,end_index+1]
            start_index = [item for item in token_offset if item[0] == pos_tok[0]]
            end_index = [item for item in token_offset if item[1] == pos_tok[1]]
            start = token_offset.index(start_index[0])
            end = token_offset.index(end_index[0]) + 1
            # add the mention's position compare to original list of all tokens for each annotated token into the tmp list
            dict_mention["mention"] = [start, end]
            mentioned_word = " ".join(dict_phrase["sentence"][start:end])

            # list URI of positive candidates by comparing the string
            list_id_pos_concept = [item[0] for item in list_id_tok if
                                   item[1].lower() == string_sentence[pos_tok[0]:pos_tok[1]]]

            # list position of each annotated concept in the phrase(i.e. positive candidates) when we search in dictionary
            list_positive_can = []
            for i in list_id_pos_concept:
                # search index for each positive candidate in the dictionary
                position_pos_can = voca_ent.word2id.get(i)
                list_positive_can.append(position_pos_can)
            # tmp list of index for positive candidates
            dict_mention["positives"] = list_positive_can
            # check if there is at least one positive candidate

            dict_mention["negatives"] = []

            # if there is a gold label for each mention
            if sentence['word'].lower() == mentioned_word.lower():
                dict_mention["entity"] = voca_ent.word2id.get(sentence['concept_id'])
            # No gold label , -1 by default
            elif sentence['word'].lower() != mentioned_word.lower():
                dict_mention["entity"] = -1

            dict_phrase["mentions"].append(dict_mention)

        # check if there is at least one mention in a sentence
        if len(dict_phrase["mentions"]) > 0:
            list_line_json.append(dict_phrase)

    # remove duplicated dictionary from list
    new_list_line_json = [x for n, x in enumerate(list_line_json) if list_line_json.index(x) == n]

    my_dict = {}
    for row in new_list_line_json:
        key = ' '.join(row['sentence'])  # use sentence as key
        if key in my_dict:
            my_dict[key]['mentions'].extend(row['mentions'])
        else:
            my_dict[key] = row

    ready_list_line_json = list(my_dict.values())

    for line_json in ready_list_line_json:
        with open(output_json, 'a') as f:
            json.dump(line_json, f)
            f.write('\n')

#json_test_file("561070.txt", "561070.ann", "el_test.json")
