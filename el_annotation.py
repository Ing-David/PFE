import json
import random
import re
<<<<<<< HEAD
from agrovoc import SKOSAnnotator
=======
from annotation import Agrovoc
>>>>>>> abb8253131c9c7d8863a38d3f576564009b5b69d
from vocabulary import Vocabulary
import pandas as pd
import en_core_web_md
from tqdm.autonotebook import tqdm

# directory of data
datadir = 'data/agrovoc'
<<<<<<< HEAD
# agrovoc's rdf
agrovoc = SKOSAnnotator(lang="en")
=======
>>>>>>> abb8253131c9c7d8863a38d3f576564009b5b69d


def fichier_json(text_string, entities_id_list, voca_ent, agrovoc):
    """
    Function to generate json file by input raw text
    :param text_string: input text string
    :param entities_id_list: List id of concepts/entities
    :param voca_ent: vocabulary of the concepts
    :param agrovoc: Agrovoc thesaurus
    """
    
    # list of sentences
    sentences = re.split(r'(?<=\.)\s+(?=[a-zA-Z])', text_string)
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
    count_json_line = 0
    for dict_phrase in sents_dict:
        output += json.dumps(dict_phrase) + "\n"
        count_json_line += 1

    return output, count_json_line


def csv_to_json(fichier_csv, name_file_json, limited_line):
    '''
    Function to convert from csv that are already extracted to json file for training
    :param fichier_csv: File csv of Agritrop
    :param name_file_json: Name of json file
    :param limited_line: Number of limited line in json file
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
    current_line = 0

    sentence_progress_bar = tqdm()
    publication_progress_bar = tqdm(file_read.iterrows())

    for line, column in publication_progress_bar:

        publication_progress_bar.set_description("Processing %dth publication" % line)
        # check if body_grobid is not an empty string
        if column['body_grobid'] != "":
            value, line_publication = fichier_json(column['body_grobid'], entIdList, voca_ent, agrovoc)

        # check if the number of sentences in each publication is less than the limited lines
        if line_publication < limited_line:
            # check if current lines are less than the limited lines
            if current_line < limited_line:
                # check if the sum of current lines and new lines from a publication are less than limited lines
                if current_line + line_publication < limited_line:
                    jsons.append(value)
                    list_value = value.splitlines()
                    current_line += len(list_value)
                # if not, fill some lines from new lines of a publication to current lines to match with the limited
                # lines
                else:
                    list_value = value.splitlines()
                    limited_sentence = list_value[:limited_line - current_line]
                    limited_value = "\n".join(limited_sentence)
                    jsons.append(limited_value)
                    current_line += len(limited_sentence)

        else:
            # check if current lines are less than the limited lines then fill some lines to match with limited line
            if current_line < limited_line:
                list_value = value.splitlines()
                limited_sentence = list_value[:limited_line - current_line]
                limited_value = "\n".join(limited_sentence)
                jsons.append(limited_value)
                current_line += len(limited_sentence)

        # description for sentences
        sentence_progress_bar.set_description(
            "Processing the %dth sentence out of %d sentences" % (current_line, limited_line))

        # current lines equal limited lines then break
        if current_line == limited_line:
            break

    if current_line < limited_line:
        print("Out of publications to produce sentences.")
        print("The total number of sentences in json file is %d" % current_line)

    else:
        print("Program Finished.")

    # write into json file
    with open(name_file_json, 'w') as json_file:
        json_file.writelines(jsons)

csv_to_json("corpus_titres_abstracts_corps_eng_articles-type_1_2_1000_limit.csv", 'el_annotated_170k.json', 170000)
