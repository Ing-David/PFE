from logging import getLogger

from rdflib import Graph, Namespace

from pyclinrec.dictionary import StringDictionaryLoader
from pyclinrec.recognizer import IntersStemConceptRecognizer

logger = getLogger("Agrovoc")


def generate_brat_normalization_database(string_entries, target_file="brat_norm_db.txt", remove_uris=True):
    concept_dictionary = {}
    for item in string_entries:
        if remove_uris:
            key = item[0].split("/")[-1].split("_")[1]
        else:
            key = item[0]
        if key not in concept_dictionary:
            concept_dictionary[key] = []
        concept_dictionary[key].append(item[1])
    with open(target_file, "w") as target_handler:
        for key in concept_dictionary.keys():
            values = concept_dictionary[key]
            entry = f"{key}"
            for value in values:
                entry += f"\tname:Name:{value}"
            entry += '\n'
            target_handler.write(entry)


class SKOSAnnotator:
    def __init__(self, graph: Graph = None, thesaurus_path="data/agrovoc_2021-03-02_core.rdf", skos_xl_labels=True,
                 lang="fr"):

        if graph is None:
            self.graph = Graph()
        else:
            self.graph = graph
        logger.info("Loading thesaurus... [{}]".format(thesaurus_path))

        self.graph.load(thesaurus_path)

        string_entries = []

        if skos_xl_labels:
            query = f"""SELECT ?x ?lf WHERE {{
                ?x a skos:Concept;
                skosxl:prefLabel ?l.
                ?l skosxl:literalForm ?lf.
                FILTER(lang(?lf)='{lang}')
            }}
            """
            pref_labels = self.graph.query(query, initNs={'skos': Namespace("http://www.w3.org/2004/02/skos/core#"),
                                                          'skosxl': Namespace("http://www.w3.org/2008/05/skos-xl#")})
        else:
            query = f"""SELECT ?x ?lf WHERE {{
                 ?x a skos:Concept;
                 skos:prefLabel ?lf.
                 FILTER(lang(?lf)='{lang}')
             }}
             """
            pref_labels = self.graph.query(query, initNs=dict(skos=Namespace("http://www.w3.org/2004/02/skos/core#")))

        for result in pref_labels:
            string_entries.append((str(result[0]), str(result[1])))

        if skos_xl_labels:
            query = f"""SELECT ?x ?lf WHERE {{
                ?x a skos:Concept;
                skosxl:prefLabel ?l.
                ?l skosxl:literalForm ?lf.
                FILTER(lang(?lf)='{lang}')
            }}
        """
            alt_labels = self.graph.query(query, initNs=dict(skos=Namespace("http://www.w3.org/2004/02/skos/core#"),
                                                             skosxl=Namespace("http://www.w3.org/2008/05/skos-xl#")))
        else:
            query = f"""SELECT ?x ?lf WHERE {{
            ?x a skos:Concept;
            skos:altLabel ?lf.
            FILTER(lang(?lf)='{lang}')
        }}
        """
            alt_labels = self.graph.query(query, initNs=dict(skos=Namespace("http://www.w3.org/2004/02/skos/core#")))

        for result in alt_labels:
            string_entries.append((str(result[0]), str(result[1])))
        generate_brat_normalization_database(string_entries, remove_uris=True)
        dictionary_loader = StringDictionaryLoader(string_entries)
        dictionary_loader.load()

        if lang == 'fr':
            self.concept_recognizer = IntersStemConceptRecognizer(dictionary_loader, "data/stopwordsfr.txt",
                                                                  "data/termination_termsfr.txt")
        else:
            self.concept_recognizer = IntersStemConceptRecognizer(dictionary_loader, "data/stopwordsen.txt",
                                                                  "data/termination_termsen.txt")

        self.concept_recognizer.initialize()

    def find_keyword_matches(self, keyword):
        _, _, matching_annotations = self.concept_recognizer.annotate(keyword)
        return_annotations = set()
        for matching_annotation in matching_annotations:
            delta = matching_annotation.end - matching_annotation.start
            if len(keyword) == delta:
                return_annotations.add((matching_annotation.concept_id, matching_annotation.matched_text,
                                        matching_annotation.start, matching_annotation.end))
        return return_annotations

    def annotate_text(self, text):
        return self.concept_recognizer.annotate(text)
