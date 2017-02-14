import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from json import (loads)
from os import path
from managers.filesys_manager import (FilesysManager)
from celery import (Celery)
from parallel_worker import process_doc_by_dois
from pymongo import MongoClient

#Load configurations
options = loads(open('data/main.cfg', 'rb').read())
db = options['db']
client = MongoClient()

#Directory setup
fm = FilesysManager()
pdf_files_dir, html_files_dir, logs_dir, doi_pdf_map, doi_fail_log = fm.dir_setup(
path.dirname(path.abspath(__file__)), options['files_dir'], options['logs_dir'])

#Create batches of documents to run on
docs_per_batch = 10000
prev_i = 0

#Compute documents in batches
if options['do_save_papers']:
  file_locs = os.listdir(pdf_files_dir) + os.listdir(html_files_dir)
  print 'Computing ' + str(len(file_locs)) + ' documents in batches of ' + str(docs_per_batch)
  for i in range(docs_per_batch, len(file_locs), docs_per_batch):
    batch_locs = file_locs[prev_i:i]
    process_doc_by_dois.delay(file_locs=batch_locs, pid=i)
    prev_i = i
  print 'Processed ' + str(len(file_locs) / docs_per_batch) + ' batches!'
elif options['do_extract_recipes']:
  db_dois = [p['doi'] for p in client[db][options['paper_collection']].find({},{'doi':True})]
  print 'Computing ' + str(len(db_dois)) + ' documents in batches of ' + str(docs_per_batch)
  for i in range(docs_per_batch, len(db_dois), docs_per_batch):
    batch_dois = db_dois[prev_i:i]
    process_doc_by_dois.delay(db_dois=batch_dois, pid=i)
    prev_i = i
  print 'Processed ' + str(len(db_dois) / docs_per_batch) + ' batches!'
