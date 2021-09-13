import logging

# -k
# 2ae951bf-2d83-42a7-944a-295a843d3307
# -u
# "http://localhost:8081/"
from annotation.agrovoc import SKOSAnnotator


def query_agroportal(base_uri, apikey, acronym, text):
    data = {
        'apikey': apikey,
        'format': 'brat',
        'ontologies': acronym,
        'longest_only': 'true',
        'text': text.replace("\n", " ")
    }
    response = requests.get(base_uri + '?' + urlencode(data))
    if response.status_code < 400:
        return response.text
    else:
        logger.warning(f"Invalid status code returned by annotator: ({response.status_code},{response.text}).")
        return ""


def annotate_local(annotator: SKOSAnnotator, text):
    brat_ann = ""
    _, _, annotations = annotator.annotate_text(text)
    annotations = sorted(annotations, key=lambda x: x.start)
    current_annotation_index = 1
    start_prev = -1
    end_prev = 0
    for annotation in annotations:
        if not (start_prev == annotation.start and end_prev == annotation.end):
            if (start_prev == annotation.start and end_prev != annotation.end) or (
                    start_prev != annotation.start and end_prev == annotation.end):
                annotation_type = "SubConcept"
            else:
                annotation_type = "Concept"
            brat_ann += f"T{current_annotation_index}\t{annotation_type} {annotation.start} {annotation.end}\t{annotation.matched_text}" + "\n"
        final_id = annotation.concept_id.split("/")[-1].split("_")[1]
        brat_ann += f"N{current_annotation_index}\t Reference T{current_annotation_index} Agrovoc:{final_id}\t{list(annotation.concept.labels)[0]}" + "\n"
        start_prev = annotation.start
        end_prev = annotation.end
        current_annotation_index += 1
    return brat_ann


import argparse
from pathlib import Path
from urllib.parse import urlencode

import pandas
import requests

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='test_corpus_generate_candidates.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

logger.info("Logging to test_corpus_generate_candidates.log")

parser = argparse.ArgumentParser(description='Generate candidate entity linking annotations for ISSA')

parser.add_argument("--agroportal", '-a', type=bool, default=False)

parser.add_argument('--apikey', '-k', type=str, nargs=1,
                    help='Agroportal API-key to use if --agroportal is selected', dest='api_key', required=False)

parser.add_argument('--agroportal-uri', '-u', type=str, nargs=1,
                    help='Base URL for Agroportal REST API', dest='base_uri', required=False)

parser.add_argument('--dump', '-d', type=str, nargs=1, help="Path to a SKOS RDF Resource")

parser.add_argument('--corpus', '-c', type=str, nargs=1,
                    help='Path to the CSV corpus to use', dest='corpus', required=True)

parser.add_argument('--output-dir', '-o', type=str, nargs=1,
                    help='Path to the CSV corpus to use', dest='output_dir', default=['test_corpus_brat'])

parser.add_argument('--limit', '-l', type=int, nargs=1, help="The number of documents to consider", dest='limit',
                    default=[10])

args = parser.parse_args()

if args.agroportal:
    api_key = args.api_key[0]
    base_uri = args.base_uri[0]
else:
    dump_path = args.dump[0]
    annotator = SKOSAnnotator(thesaurus_path=dump_path)
corpus_in = args.corpus[0]
output_dir = args.output_dir[0]
limit = args.limit[0]

logger.info('Loading CSV corpus from ' + str(corpus_in))
corpus_df = pandas.read_csv(corpus_in)
# corpus_df['body_grobid'] = corpus_df['body_grobid'].astype('str')

current_count = 0

output_dir_path = Path(output_dir)
if not output_dir_path.exists():
    output_dir_path.mkdir()

for index, row in corpus_df.iterrows():
    text = row['body_grobid']
    if len(text) > 0 and text != 'nan':
        id = row['CLE']
        output_annotations = ""
        if args.agroportal:
            logger.debug("Querying Agroportal...")
            output_annotations = query_agroportal(base_uri, api_key, 'AGROVOC', text)
            logger.debug("Received: " + output_annotations)
        else:
            output_annotations = annotate_local(annotator, text)
        with open(output_dir_path / f'{id}.txt', 'w') as text_file:
            text_file.write(text)
        with open(output_dir_path / f'{id}.ann', 'w') as annot_file:
            annot_file.write(output_annotations)
        current_count += 1

    if current_count >= limit:
        logger.info(f'Limit of {limit} reached...')
        break
