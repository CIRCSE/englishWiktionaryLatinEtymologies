# englishWiktionaryLatinEtymologies

## Overview

This repository contains a pipeline that extracts etymological information from Wiktionary and converts it into RDF triples. The resulting data models etymological relationships between lexical items and their ancestors using Semantic Web standards. The pipeline processes Wiktionary data, reconstructs etymological chains, and serializes them into RDF. It is designed as language-agnostic, but it pays special attention to Latin, since that's the dataset that was first produced with it.



## Contents
- WiktionaryEtymologiesToRDF.py  
  Main pipeline script implementing the extraction, transformation, curation, enrichment, and RDF serialization workflow.

- englishWiktionaryLatinEtymologies.nt  
  RDF dataset serialized in N-Triples format. More machine-oriented. Allows for processing massive datasets without having problems with RAM.

- englishWiktionaryLatinEtymologies.ttl  
  RDF dataset serialized in Turtle format. More human-readable.

## Purpose
The goal of this project is to produce a structured representation of etymological chains extracted from Wiktionary that can be consumed by tools and systems that operate on RDF knowledge graphs.

## License
The dataset is released under the same license as the underlying Wiktionary data (Creative Commons Attribution-ShareAlike).
