# PFE
# dl4el: A distant learning approach to entity linking
A Python implementation of ACL2019 paper

[1] Phong Le and Ivan Titov. [Distant Learning for Entity Linking with Automatic Noise Detection](https://arxiv.org/abs/1905.07189). ACL 2019.

# Introduction

We implement this method by using the knowledge base [Agrovoc](https://agrovoc.fao.org/agrovocReleases/agrovoc_2021-05-04_core.rdf.zip) in the RDF format and the raw text for training set(i.e. non annotated corpus) from [Agritrop](https://agritrop.cirad.fr/) of Cirad. We used the scientific publications in pdf format. Here is the example of a [pdf format](http://agritrop.cirad.fr/557447/1/document_557447.pdf). In order to extract the raw text from the pdfs format, we use the tools [CERMINE](https://github.com/CeON/CERMINE) and [GROBID](https://github.com/kermitt2/grobid).

# Installation

There are some modules that you need to install. If you use `pip` in your environment using the following commands:

`!pip install rdflib`

`!pip install git+https://github.com/twktheainur/pyclinrec.git`

`!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.0.0/en_core_web_md-3.0.0.tar.gz`

`!python -c 'import nltk; nltk.download("stopwords"); nltk.download("punkt" )'`

# Training Data

In order to generate the training data in the google colab

First, you need to clone the repositoty and go into the master branch by using the commad line:

`!git clone https://github.com/Ing-David/PFE.git && cd PFE && git checkout master`

Create a folder to store all your requirement data(i.e. yourfolder/data/..) Here my main folder is `PFE`:

`!mkdir -p PFE/data/{glove,agrovoc/el_annotation,EL}`

Download the RDF format of agrovoc and move it into `PFE/data` by using the command line:

`!wget https://agrovoc.fao.org/agrovocReleases/agrovoc_2021-05-04_core.rdf.zip && unzip agrovoc_2021-05-04_core.rdf.zip && mv agrovoc_2021-05-04_core.rdf PFE/data/agrovoc_2021-03-02_core.rdf`

Go into the folder `PFE/data/agrovoc` and unzip agrovoc's dictionary by using the command line:

`!cd PFE/data/agrovoc && tar -xjf agrovoc_files.tar.bz`

To generate the training set based on Agritrop corpus using the command line:

`!cd PFE && python el_annotation.py`

Move json file of training set into `data/el_annotation` consider we are in the folder `PFE` by using the command:

`mv el_annotated.json PFE/data/el_annotation/el_annotated.json`

# Glove data

When we are in the folder `PFE`, download the glove's file, unzip it by using the command line:

`!wget http://nlp.stanford.edu/data/glove.840B.300d.zip && unzip glove.840B.300d.zip`

Generate the file `glove.word` and `word_embeddings.npy` by using the command line in colab:

`!cd PFE && python glove.py`

Move glove's data into folder glove. Consider we are in folder `PFE` by using the command line:

`mv glove.word PFE/data/glove/glove.word`

`mv word_embeddings.npy PFE/data/glove/word_embeddings.npy`

# Training data

To train, using the command line:

`!cd PFE && python el_main.py` 




