import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from json import (loads)
from os import (path)
from classifiers.synth_para_classifier import (SynthParaClassifier)
from extractors.info_extractor import (InfoExtractor)
from managers.filesys_manager import (FilesysManager)
from managers.download_manager import (DownloadManager)
import spacy
from sys import (argv, stdout)
from celery import (Celery)

#Setup Celery app
app = Celery('parallel_worker', backend='rpc://', broker='amqp://')
app.config_from_object('celeryconfig')


### PARALLEL PIPELINE DEFINITION START ###
@app.task
def process_doc_by_dois(file_locs=[], db_dois=[], pid=None):
  '''
  Processes a set of documents through the entire pipeline, given their DOIs. Assumes that we already have the PDFs.
  '''

  nlp = spacy.load('en')

  #Load configurations
  options = loads(open('data/main.cfg', 'rb').read())
  db = options['db']

  dm = DownloadManager(db)
  #Directory setup
  fm = FilesysManager()
  pdf_files_dir, html_files_dir, logs_dir, doi_pdf_map, doi_fail_log = fm.dir_setup(
  path.dirname(path.abspath(__file__)), options['files_dir'], options['logs_dir'])

  spc = SynthParaClassifier(db)
  spc.load('bin/synthesis_para_classifier.pkl')

  #Throw the files into Paper() objects and save them
  if options['do_save_papers']:
    dm.save_papers(pdf_files_dir, html_files_dir, doi_pdf_map, options['paper_collection'], file_locs=file_locs, overwrite=False, para_classifier=spc)
  elif options['do_extract_recipes']:
    ex = InfoExtractor(db)
    ex.load_nlp(nlp)
    max_papers = 0 #0 = no limit
    ex.load_all_papers(max_papers, skip=0, collection=options['paper_collection'], dois=db_dois)
    ex.extract_and_save_all_papers(overwrite=True)
