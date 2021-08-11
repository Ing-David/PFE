import json
from annotation import Agrovoc
from vocabulary import Vocabulary
import en_core_web_md
from nltk.tokenize import word_tokenize
from collections import defaultdict
from operator import itemgetter
import re
from tqdm.notebook import tqdm
tqdm.pandas()

# directory of data
datadir = 'data/agrovoc'
# spacy with eng
nlp = en_core_web_md.load()


def add_uri(id_number):
    """
    Function to add the domain name of agrovoc
    :param id_number: The URI of a concept
    """

    str_id = str(id_number)
    s = "http://aims.fao.org/aos/agrovoc/c_" + str_id
    return s


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
      'sentence_id': 0,
      'start': 13,
      'end': 18},
     {'standoff_id': 2,
      'entity_type': 'Concept',
      'offset_start': 26,
      'offset_end': 30,
      'word': 'corn',
      'concept_id': 'http://aims.fao.org/aos/agrovoc/c_12332',
      'original_sentence': 'Ceci est une wheat phrase corn.\n',
      'sentence_id': 0,
      'start': 26,
      'end': 30},
     {'standoff_id': 3,
      'entity_type': 'Concept',
      'offset_start': 61,
      'offset_end': 67,
      'word': 'barley',
      'concept_id': 'http://aims.fao.org/aos/agrovoc/c_823',
      'original_sentence': 'Ceci est une deuxième phrase barley.',
      'sentence_id': 1,
      'start': 29,
      'end': 35}]
    """


    # looping through .txt files
    with open(text_file) as f:
        list_phrase = f.readlines()
        f.seek(0)
        contents = f.read()
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
                    (sum(i for i in list_length_phrase[0:index]), sum(i for i in list_length_phrase[0:index + 1])))

    # looping through .ann files in the data directory
    STANDOFF_ENTITY_PREFIX = 'T'
    STANDOFF_RELATION_PREFIX = 'N'
    entities = []
    relations = []
    # process .ann file - place entities and relations into 2 seperate lists of tuples
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
                entity['word'] = standoff_line[4]
                entities.append(entity)

            elif standoff_line[0][0] == STANDOFF_RELATION_PREFIX:
                relation = {}
                relation['standoff_id'] = int(standoff_line[0][1:])
                relation['concept_id'] = standoff_line[3].split(":")[1]
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
            j = word_tokenize(phrase)
            if contents[dict_concept['offset_start']:dict_concept['offset_end']] in j and (
                    offset_phrase[index][0] < dict_concept['offset_start'] and dict_concept['offset_end'] <
                    offset_phrase[index][1]):
                dict_concept['original_sentence'] = list_phrase[index]

    # loop to get the sentence id for each concept
    for index_phrase, phrase in enumerate(list_phrase):
        for dictionary_concept in list_dict_concept:
            if (dictionary_concept['original_sentence'] == phrase):
                dictionary_concept['sentence_id'] = index_phrase

    # loop to add URI complete to each mention
    for concept_dict in list_dict_concept:
        concept_dict["concept_id"] = add_uri(concept_dict["concept_id"])

    # loop to get the offset of each mention for individual sentence
    for con_dict in list_dict_concept:
        text = con_dict["original_sentence"]
        tokens = [con_dict['word']]
        pattern = re.compile(
            fr'(?<!\w)(?:{"|".join(sorted(map(re.escape, tokens), key=len, reverse=True))})(?!\w)(?!\.\b)', re.I)
        offsets = individual_tokenOffset(text, pattern)
        con_dict["start"] = offsets[0]["IndividualOffsetBegin"]
        con_dict["end"] = offsets[0]["IndividualOffsetEnd"]

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
    entIdList = list(ent2nameId.keys())
    with open(text_file) as f:
        list_phrase = f.readlines()
        f.seek(0)
        contents = f.read()

    text = nlp(contents)
    sentences = list(text.sents)

    list_dict_concept = list_dict_mention(text_file, brat_file)

    # loop through each phrase
    for index, sentence in tqdm(enumerate(sentences)):
        dict_phrase = {}
        string_sentence = str(sentence)
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
        # position of each annotated token in each phrase
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
            # list id of positive candidates by comparing the string
            list_id_pos_concept = [item[0] for item in list_id_tok if item[1] == string_sentence[pos_tok[0]:pos_tok[1]]]
            # list position of each annotated concept in the phrase(i.e. positive candidates) when we search in dictionary
            list_positive_can = []
            for i in list_id_pos_concept:
                # search index for each positive candidate in the dictionary
                position_pos_can = voca_ent.word2id.get(i)
                list_positive_can.append(position_pos_can)
            # tmp list of index for positive candidates
            dict_mention["positives"] = list_positive_can
            # check if there is at least one positive candidate
            if len(list_positive_can) > 0:
                dict_mention["negatives"] = []
                for dict_concept in list_dict_concept:
                    #  check which sentence belongs to a mention and check if the offset start and end match
                    if dict_concept["sentence_id"] == index and annotated_concepts[0]["start"] == dict_concept[
                        "start"] and annotated_concepts[0]["end"] == dict_concept["end"]:
                        # gold entity from BRAT
                        dict_mention["entity"] = voca_ent.word2id.get(dict_concept["concept_id"])
            dict_phrase["mentions"].append(dict_mention)

        # check if there is at least one mention in a sentence
        if len(dict_phrase["mentions"]) > 0:
            # write into Temperary dict_phrase into json file
            with open(output_json, 'a') as f:
                json.dump(dict_phrase, f)
                f.write('\n')

# json_test_file("test.txt", "test.ann", "test.json")