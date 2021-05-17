# PFE
# dl4el: A distant learning approach to entity linking
A Python implementation of ACL2019 paper

[1] Phong Le and Ivan Titov. [Distant Learning for Entity Linking with Automatic Noise Detection](https://arxiv.org/abs/1905.07189). ACL 2019.

# Introduction

We implement this method by using the knowledge base [Agrovoc](https://agrovoc.fao.org/agrovocReleases/agrovoc_2021-05-04_core.rdf.zip) in the RDF format and using the raw text for training set(i.e. non annotated corpus) from [Agritrop](https://agritrop.cirad.fr/) of Cirad. We used the scientific publications in pdf format. Here is the example of a [pdf format](http://agritrop.cirad.fr/557447/1/document_557447.pdf). In order to extract the raw text from the pdfs format, we use the tools [CERMINE](https://github.com/CeON/CERMINE) and [GROBID](https://github.com/kermitt2/grobid).

# Usage




