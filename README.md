# The Synthesis Project - Database

The Synthesis Project aims to catalog and predict materials synthesis routes. This repository contains code for populating a materials synthesis database from a corpus of PDF/HTML journal articles. The database is exposed to the public via the Synthesis Project API (not contained in this repository).

This repository includes the following:

+ Higher-level scripts and wrappers for downloading a corpus of articles (with dependencies on `olivettigroup/article-downloader`)
+ Objects and scripts for extracting text and parsing / entity recognition / etc. for materials science text
+ A script for packaging the extracted database and sending it to an API-serving endpoint

In general, **run everything from the `synthesisdatabase` folder as the working directory, not the repository root**. This includes the run scripts, tests, etc.

## Dependencies
Naturally, you need everything in `requirements.txt`, so use `pip install -r requirements.txt` to install all the dependencies. You also need a MongoDB server (e.g., have the daemon `mongod` running) and you'll need to have the `iesl/watr-works` repo (which is a public repo) installed for extracting text from PDF files.

If you didn't already have [spacy](https://spacy.io/docs) installed before running this, you need to also run `python -m spacy.en.download all`.

In order to download article files from publisher APIs, you need the appropriate API keys, set to environment variables `ELS_API_KEY` and `CRF_API_KEY`, for Elsevier and CrossRef, respectively. You'll also need a Materials Project API key stored in the environment variable `MAPI_KEY`, and a Citrination API key stored in `CITRINATION_API_TOKEN` to retrieve materials properties.

## Conceptual Overview

The pipeline, at a high level, does the following:

0. Downloads PDFs from publisher APIs based on a set of search queries, using the `article-downloader` module (see [external link](https://www.github.com/olivettigroup/article-downloader)).
0. Consumes JSON files produced by `iesl/watr-works`, which contain processed plaintext from the PDFs. (or HTMLs from publishers)
0. Classifies relevant paragraphs within papers for further synthesis parameter extraction.
0. Extracts synthesis parameters from papers. Also extracts additional info (e.g., properties, morphologies) from other sections of papers.
0. Builds a map of materials properties based on extracted materials from papers, using API calls to external databases.

The outputs of the pipeline are all saved to MongoDB. The schema of saved objects are outlined in `base_models.py` and `models.py`.

## Testing

Run `nosetests -w synthesisdatabase`. Make sure you have nose installed. Automated testing is done through CircleCI.

## Running (parallel)

The parallel batch-processing approach uses `celery`. In order to run using celery workers, you'll need `rabbitmq` installed, and you'll have to be running `rabbitmq-server` in the background. Then, execute `shell_scripts/parallel_worker_start.sh` to boot up the workers. After that, run `shell_scripts/run_parallel_pipeline.sh` to send tasks to the message queue. The workers will automatically read and execute tasks from the queue. You can also run `shell_scripts/flower_start.sh` to start up a Flower monitoring server. The behavior of each worker is governed by `parallel_worker.py`.

Parallel workers will log to the `logs/` directory, with files named in the pattern `parallel-*.log`.

## Running (non-parallel)

Run `main.py` to go through the whole routine in a single thread. In practice, the **non-parallel version must be used for downloading PDFs and anything else that involves API calls**, since parallelism breaks API rate limits.

A good way to run the main pipeline on a server is the following command:

    nohup python main.py -c data/main.cfg &

## Accessing run outputs

After the pipeline runs successfully,
    1. Type `mongo` into command line
    2. Type `show dbs` into mongo command line
    3. Check `predsynth` exists, and `use predsynth`
    4. `predsynth` should be populated with `papers` collection, and contain objects in that collection.
    5. To check: `show collections`

## Other running guidelines

`nohup` is only relevant if you're SSHed into a server and don't want disconnects to kill your run. You can pass in config files using the `-c` flag.

## Configuration

You can find config files in the `data/` directory -- they are JSON files that dictate which 'sections' of the pipeline should be enabled. An important note is that **while you can use the same config file format for `main.py` and `parallel_worker.py`, they don't check for exactly the same options.**

Config files should look something like the following...

    {
      "mp_queries":                           "data/mp_queries.json",
      "cr_queries":                           "data/cr_material_queries.json",
      "do_mp_queries":                        false,
      "do_cr_queries":                        false,
      "do_save_papers":                       false,
      "do_extract_recipes":                   true,
      "paper_collection":                     "papers",
      "db":                                   "predsynth",
      "files_dir":                            "/raid10/synthesis-project/files/",
      "logs_dir":                             "logs/"
    }


mp_queries: queries to search the materials project database. Will return articles based on the given material query.
cr_queries: queries to search crossref database. Will return articles based on the given query.
do_save_papers: save papers from raw HTML/PDF files to the DB
do_extract_recipes: extract entities/etc from text in papers and save to DB

A config file for the main pipeline must be __valid JSON__ and it essentially tells the pipeline which processing steps to 'turn on' and some basic database configurations.

The `mp_queries` and `dl_queries` fields should point to file locations that are __valid JSON__ documents which are just a list of queries. For example:

    [
      "battery+synthesis",
      "electrode+synthesis",
      "lithium+battery",
      "lithium+electrode+synthesis",
      "fuel+cell+synthesis",
      "solar+synthesis",
      "photovoltaic+synthesis",
      "supercapacitor+synthesis",
      "nanomaterial+nanostructured+synthesis"
    ]

The `files_dir` directory is where all the PDFs and converted JSONs get stored, along with metadata files. Additional metadata files will get stored to the `logs_dir` directory.
