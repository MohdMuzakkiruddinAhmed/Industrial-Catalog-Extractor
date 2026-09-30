# Data availability

## Source catalogs

The industrial catalog PDFs used during development and validation are not included
in this repository. They may be copyrighted and the Apache-2.0 code license does
not grant permission to redistribute, extract, publish, or embed them.

The reported inventory contained 110 vendor product catalogs with 6,069 pages and
110 industrial reference documents with 11,768 pages. These aggregate counts are
provided to describe experiment scale, not to distribute the corpus.

## Using your own data

Place PDFs that you are authorized to process under a corpus directory and update
`paths.corpus_root` in a local copy of the YAML configuration. Do not commit that
local config, the PDFs, raw model responses, SQLite databases, or embeddings.

Before processing third-party catalogs, consider:

- copyright and database rights;
- confidentiality and contractual restrictions;
- personal contact information;
- export-control or controlled technical data;
- retention and deletion requirements;
- whether a remote endpoint is permitted to receive rendered pages.

## Public synthetic fixture

`examples/create_sample_catalog.py` generates a fictional two-page PDF whose names,
identifiers, and specifications were created solely for this repository. It is the
public test fixture for pipeline orchestration and citation behavior.

The fixture does not measure real model accuracy or represent an actual manufacturer.

## Public experiment artifacts

Safe aggregate summaries, configuration templates, schema versions, code, tests,
and documentation can be published. Raw page elements, screenshots, provider
responses, products, vector databases, and embeddings can reproduce source content
and should not be published without a separate rights and privacy review.

The public-safe aggregate evidence and observed server profile are available in
machine-readable form under `reproducibility/`. These files are transcriptions of
verified private operational artifacts, not raw run exports or a corpus release.
