from agrovoc import Agrovoc
from rdflib import Graph
import spacy
import json
import random
import pandas as pd

nlp = spacy.load('en_core_web_md')
agrovoc = Agrovoc(lang="en")

def fichier_json (text_string, name_file_json, limited_line):
    """
    function to generate json file by input raw text
    arg text_string: input text string
    arg name_file: input the name for the json file
    arg limited_line: input how many lines to write into the json file
    """
    # access dictionary of agrovoc with correspond id for searching the index location of id for each candidate(positive and negative)
    df = pd.read_csv("data/agrovoc/agrovoc-entity.tsv", sep="\t", header=None)
    text_doc = nlp(text_string)
    sentences = list(text_doc.sents)
    #loop through each phrase
    for sentence in sentences:
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
                # add the mention's position compare to original list of all tokens for each annotated token into the tmp list
                dict_mention["mention"] = [start, end]
                # list id of positive candidates by comparing the string
                list_id_pos_concept = [item[0] for item in list_id_tok if item[1] == string_sentence[pos_tok[0]:pos_tok[1]]]
                # list position of each annotated concept in the phrase(i.e. positive candidates) when we search in dictionary
                list_positive_can = []
                for i in list_id_pos_concept:
                    # search index for each positive candidate in the dictionary
                    position_pos_can = int(df[df[0] == i].index.values)
                    list_positive_can.append(position_pos_can)
                # tmp list of index for positive candidates
                dict_mention["positives"] = list_positive_can
                # before generating the negative candidate we need to make sure that there is at one positive candidate
                if len(list_positive_can) > 0:
                    list_neg_can = []
                    # generate 10 negative candidates
                    for i in range(0, 10):
                        # random all concepts id in the dictionary
                        n = random.randint(0, 38610)
                        # avoid positive concepts
                        if n not in [j for j in list_positive_can]:
                            # avoid negative concepts that already existed in the list
                            if n not in [k for k in list_neg_can]:
                                list_neg_can.append(n)
                    dict_mention["negatives"] = list_neg_can
                dict_phrase["mentions"].append(dict_mention)
        with open(name_file_json, 'a') as f:
            num = sum(1 for line in open(name_file_json))
            if num < limited_line:
                json.dump(dict_phrase, f)
                f.write('\n')
            else:
                break

'''
integral_text = ("Mechanical properties studies of wood usually involve destructive wood assessments, with time-consuming tests that use large amounts of resource (wood). Although this is not a limiting factor, it could be attenuated by the use of a nondestructive technique known as near infrared spectroscopy (NIRS). This technique has been applied to evaluate compounds containing C-H, N-H, S-H or O-H bonds, and involves quick analyses and can be applied to process control tasks. The objective of this work is to use the NIRS technique to obtain calibrations for mechanical properties of Eucalyptus sp. wood. A natural E. urophylla hybrid at age 7 was used as obtained from V&M Florestal crops. Spectra were measured directly in solid wood (radial, tangential and transverse faces) and in ground wood, in diffuse reflectance mode, using a Bruker spectrometer in the 800 to 1,500 nm range. The NIRS technique proved suitable to estimate modulus of elasticity in solid wood, with values r=0.91 and RPD=2.6, and in ground wood, with values r=0.87 and RPD=2.0. Modulus of rupture and compressive strength presented r values below 0.9. First and second derivative pretreatments provided a slight increase in correlation values for the properties in question. Calibrations for different plank faces did not present a defined variation pattern. Solid wood and ground wood presented similar correlation values for all properties.")
fichier_json(integral_text,"el_annotated_example.json",20)
'''