# -*- coding: utf-8 -*-
'''
created: Wed Mar 31 11:15:19 2021
last modified: Wed Mar 31 11:15:19 2021
@author: David ING
'''
import rdflib
import pandas as pd

#load rdf knowledge graph of agrovoc
g = rdflib.Graph()
g.parse("agrovoc_2021-03-02_core.rdf")

#create agrovoc entity(URI and Label) for each concept into tsv file
def agrovoc_entity():

    qres = g.query("""PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
                  PREFIX skosxl:<http://www.w3.org/2008/05/skos-xl#>
                  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                  SELECT ?entity ?label WHERE {
                  ?entity a skos:Concept.
                  ?entity skosxl:prefLabel ?pl.
                  ?pl skosxl:literalForm ?label_l.
                  FILTER langMatches( lang(?label_l), "EN" )
                  BIND (STR(?label_l)  AS ?label)
                  }""")   
    column_names = ["URI","Label"]
    df_entities = pd.DataFrame([row for row in qres], columns = column_names)
    df_entities.to_csv("agrovoc-entity.tsv",sep="\t", index=False, header=None)

#create triple(sub pred obj) for each concept in agrovoc into tsv file
def agrovoc_triples():

    qres = g.query("""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                       PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                       SELECT * WHERE {
                          ?sub ?pred ?obj .
                        }""")
    column_names = ["sub","pred","obj"]
    df_triples = pd.DataFrame([row for row in qres], columns = column_names)
    df_triples.to_csv("agrovoc-triples.tsv",sep="\t", index=False, header=None)

#create agrovoc(type of all nodes) into tsv file
def agrovoc_types():

    qres = g.query("""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                      SELECT ?type WHERE {?node a ?type.}""")
    column_names = ["type"]
    df_types = pd.DataFrame([row for row in qres], columns = column_names)
    df_types.to_csv("agrovoc-type.tsv",sep="\t", index=False, header=None)

#create agrovoc(type-instance) into tsv file
def agrovoc_type_instance():
    qres = g.query("""PREFIX skos: <http://www.w3.org/2004/02/skos/core#> 
                      PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#> 
                      SELECT ?node ?type WHERE {?node a ?type.} Limit 10""")
    column_names = ["node","type"]
    df_type_instance = pd.DataFrame([row for row in qres], columns = column_names)
    df_type_instance.to_csv("agrovoc-type-instance.tsv",sep="\t", index=False, header=None) 

#agrovoc_entity()
#agrovoc_triples()
#agrovoc_types()
#agrovoc_type_instance()

