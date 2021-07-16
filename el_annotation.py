import spacy
import json
import random
from annotation import Agrovoc
from vocabulary import Vocabulary
import pandas as pd
import en_core_web_md
from tqdm.notebook import tqdm

# directory of data
datadir = 'data/agrovoc'
# spacy with eng
nlp = en_core_web_md.load()

def fichier_json(text_string,entities_id_list, voca_ent, agrovoc):
    """
    function to generate json file by input raw text
    arg text_string: input text string
    arg name_file: input the name for the json file
    arg limited_line: input how many lines to write into the json file
    """
    # convert to text_doc
    text_doc = nlp(text_string)
    # list of sentences
    sentences = list(text_doc.sents)
    # list to store each phrase in an integral text
    sents_dict = []
    # loop through each phrase
    for sentence in sentences:
        # dictionary for each phrase
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
        # Condition for each phrase to has at least two different mentions
        if len(list_pos_tok) > 2:
            # position of each annotated token in each phrase
            for pos_tok in list_pos_tok:
                # dictionary mention
                dict_mention = {}
                # position of each mentioned token in list of all tokens [start_index,end_index+1]
                start_index = [item for item in token_offset if item[0] == pos_tok[0]]
                end_index = [item for item in token_offset if item[1] == pos_tok[1]]
                start = token_offset.index(start_index[0])
                end = token_offset.index(end_index[0]) + 1
                # add the mention's position compare to original list of all tokens for each annotated token into the
                # tmp list
                dict_mention["mention"] = [start, end]
                # list id of positive candidates by comparing the string
                list_id_pos_concept = [item[0] for item in list_id_tok if
                                       item[1] == string_sentence[pos_tok[0]:pos_tok[1]]]
                # list position of each annotated concept in the phrase(i.e. positive candidates) when we search in
                # dictionary
                list_positive_can = []
                for i in list_id_pos_concept:
                    # search index for each positive candidate in the dictionary
                    position_pos_can = voca_ent.word2id.get(i)
                    list_positive_can.append(position_pos_can)
                # tmp list of index for positive candidates
                dict_mention["positives"] = list_positive_can
                # before generating the negative candidate we need to make sure that there is at one positive candidate
                if len(list_positive_can) > 0:
                    list_neg_can = []
                    # generate 10 negative candidates
                    for i in range(0, 50):
                        # random all concepts id in the entIdList
                        m = random.sample(entities_id_list, 1)
                        # convert from list of one element to integer
                        n = int(m[0])
                        # avoid positive concepts
                        if n not in [j for j in list_positive_can]:
                            # avoid negative concepts that already existed in the list
                            if n not in [k for k in list_neg_can]:
                                list_neg_can.append(n)
                                # check if the list already contained 10 candidates 
                                if len(list_neg_can) == 10:
                                    break
                    dict_mention["negatives"] = list_neg_can
                dict_phrase["mentions"].append(dict_mention)
        # check if there is at least one mention in a sentence
        if len(dict_phrase["mentions"]) > 0:
            sents_dict.append(dict_phrase)
        # if not skip
        else:
            continue
            
    output = ""
    
    for dict_phrase in sents_dict:
        output += json.dumps(dict_phrase) + "\n"

    return output, len(sentences)


def csv_to_json(fichier_csv, name_file_json, limited_line):
    '''
    function to convert from csv that are already extracted to json file for training
    '''
    # agrovoc's rdf
    agrovoc = Agrovoc(lang="en")
    # access dictionary of agrovoc with correspond id for searching the index location of id for each candidate(
    # positive and negative)
    voca_ent, _ = Vocabulary.load(datadir + '/agrovoc-entity.tsv', normalization=False, add_pad_unk=False)
    # get the entIdList for random negative candidates
    ent2nameId = {}
    with open(datadir + '/agrovoc-entity.tsv', 'rt') as f:
        g = f.read()
        h = filter(None, g.split("\n"))
        for i in h:
            ent2nameId[voca_ent.word2id.get(i)] = i
    entIdList = list(ent2nameId.keys())

    file_read = pd.read_csv(fichier_csv)
    file_read['body_grobid'] = file_read['body_grobid'].astype('str')
    jsons = []
    sentences = 0
    for line, column in tqdm(file_read.iterrows()):
        if column['body_grobid'] != "":
            value, count = fichier_json(column['body_grobid'],entIdList, voca_ent, agrovoc)
        if sentences + count <= limited_line:
            sentences += count
            jsons.append(value)
        else:
            break
    with open(name_file_json, 'w') as json_file:
        json_file.writelines(jsons)


csv_to_json("corpus_titres_abstracts_corps_eng_articles-type_1_2_4_100_limit.csv", 'el_annotated.json', 170000)
