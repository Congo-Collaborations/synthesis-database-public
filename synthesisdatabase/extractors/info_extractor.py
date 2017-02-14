#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

from re import (sub, findall)
from json import (loads, dumps)
from bson.objectid import (ObjectId)
from pymongo import MongoClient
from models import (Paper, Material, Descriptor, Condition, Connection, Apparatus, Property, Amount, Operation, Paragraph)
from classifiers.token_classifier import TokenClassifier
from os import (environ)
from autologging import (logged, traced)
from chemdataextractor import Document as cdeDoc
from collections import Counter
import traceback
import numpy as np

@logged
class InfoExtractor(object):

  def __init__(self, db):
    self.db = db
    self.connection = MongoClient()
    self._load_token_classifier()
    self._load_lexicon()
    self._load_unit_conversions()

    self.__logger.info( self.__class__.__name__ + ' initialized' )

    self.token_class_lookups = {
      0: 'null',
      1: 'amount',
      2: 'condition',
      3: 'material',
      4: 'target',
      5: 'operation',
      6: 'descriptor',
      7: 'property',
      8: 'apparatus',
      9: 'intermediate',
      10: 'num'
    }

  def load_nlp(self, nlp):
    self.nlp = nlp

  def load_all_papers(self, limit, skip=0, collection='papers', dois=[]):
    if len(dois) == 0:
      self.papers = self.connection[self.db][collection].find()
    else:
      self.papers = list(self.connection[self.db][collection].find(
        {'doi': {'$in':dois}}, no_cursor_timeout=True
        ))
    self.__logger.info( 'Loaded all papers!' )

  def save_verbose_info(self, save_data={}, fpath='/raid10/synthesis-project/verbose_extracted_papers/'):
    safe_doi = str(save_data['doi']).translate(None, '/.()')
    with open(fpath + safe_doi + '.json', 'wb') as f:
      f.write(dumps(save_data, indent=2))

  def extract_and_save_all_papers(self, overwrite=False):
    for i, paper in enumerate(self.papers):
      if not overwrite:
        matches = list(paper['operations'])
        #Assume if a paper has operations, then it has other extracted stuff too
        if len(matches) == 0:
          self.__logger.info( "EXTRACT: (Non-Overwriting) Extracting paper #" + str(i)  + ": " + str(paper['doi']) )
          self.extract_paper(paper)
        else:
          self.__logger.info( "DUPLICT: (Non-Overwriting) NOT extracting paper #" + str(i)  + ": " + str(paper['doi']) )
      else:
        self.__logger.info( "EXTRACT: (No overwrite checks) Extracting paper #" + str(i)  + ": " + str(paper['doi']) )
        self.extract_paper(paper)

    self.__logger.info( "SUCCESS: Extracted and saved all papers!" )

  def extract_paper(self, paper, save_verbose=False):
    paper['materials'] = []
    paper['operations'] = []
    paper['connections'] = []

    mats_to_coref = []

    if save_verbose:
      verbose_data = {
        'doi': paper['doi']
      }

    #Extract data from recipes
    try:
      body_text = ''
      recipe_text = ''
      for paragraph in paper['paragraphs']:
        if paragraph['type'] == unicode('recipe'):
          recipe_text += paragraph['text'] + '\n'
        else:
          body_text += paragraph['text'] + '\n'

      cde_doc = cdeDoc(
        unicode(paper['title']) + '\n' +
        unicode(paper['abstract']) + '\n' +
        recipe_text
        )

      if len(recipe_text) > 0:
        labels = self.apply_token_labels(recipe_text, cde_doc=cde_doc)
        chunks, labels = self.get_relevant_chunks(labels)
        mats, ops, conns = self.get_recipe_from_chunks(chunks)
        mats, ops, conns = self.refine_recipe(mats, ops, conns, chunks)

        if save_verbose:
          verbose_data['recipe_chunks'] = loads(dumps(chunks, default=unicode))
          verbose_data['recipe_token_labels'] = loads(dumps(labels, default=unicode))
          verbose_data['recipe_text'] = recipe_text

        paper['materials'].extend(mats)
        paper['operations'].extend(ops)
        paper['connections'].extend(conns)
      else:
        self.__logger.info( 'SKIPPED: No recipe to extract in ' + str(paper['doi']) )
        return -1

      if len(body_text) > 0:
        labels = self.apply_token_labels(body_text, cde_doc=cde_doc)
        chunks, labels = self.get_relevant_chunks(labels)
        mats, ops, conns = self.get_recipe_from_chunks(chunks)

        if save_verbose:
          verbose_data['body_chunks'] = loads(dumps(chunks, default=unicode))
          verbose_data['body_token_labels'] = loads(dumps(labels, default=unicode))
          verbose_data['body_text'] = body_text

        mats_to_coref.extend(mats)

      self.__logger.info( 'SUCCESS: Extracted targets/operations/materials from ' + str(paper['doi']) )
    except Exception, e:
      self.__logger.warning( 'FAILURE: Failed to extract targets/operations/materials from ' + str(paper['doi']) )
      self.__logger.warning( 'ERR_MSG: ' + str(e) )


    #Extract target materials
    try:
      if 'title' in paper:
        if len(paper['title']) > 0:
          labels = self.apply_token_labels(paper['title'], cde_doc=cde_doc)
          chunks, labels = self.get_relevant_chunks(labels)
          mats, ops, conns = self.get_recipe_from_chunks(chunks)

          if save_verbose:
            verbose_data['title_chunks'] = loads(dumps(chunks, default=unicode))
            verbose_data['title_token_labels'] = loads(dumps(labels, default=unicode))
            verbose_data['title_text'] = paper['title']

          mats_to_coref.extend(mats)

          self.__logger.info( 'SUCCESS: Extracted target materials (from title) from ' + str(paper['doi']) )
    except Exception, e:
      self.__logger.warning( 'FAILURE: Failed to extract (title) target materials from ' + str(paper['doi']) )
      self.__logger.warning( 'ERR_MSG: ' + str(e) )

    try:
      if 'abstract' in paper:
        if len(paper['abstract']) > 0:
          labels = self.apply_token_labels(paper['abstract'], cde_doc=cde_doc)
          chunks, labels = self.get_relevant_chunks(labels)
          mats, ops, conns = self.get_recipe_from_chunks(chunks)

          if save_verbose:
            verbose_data['abstract_chunks'] = loads(dumps(chunks, default=unicode))
            verbose_data['abstract_token_labels'] = loads(dumps(labels, default=unicode))
            verbose_data['abstract_text'] = paper['abstract']

          mats_to_coref.extend(mats)

          self.__logger.info( 'SUCCESS: Extracted target materials (from abstract) from ' + str(paper['doi']) )
    except Exception, e:
      self.__logger.warning( 'FAILURE: Failed to extract (abstract) target materials from ' + str(paper['doi']) )
      self.__logger.warning( 'ERR_MSG: ' + str(e) )

    try:
      coreferenced_materials = []
      #cde_recs = [rec['names'] for rec in cde_doc.records.serialize() if 'names' in rec]

      for material in paper['materials']:
        modified_material = material
        if material['is_target']:
          for coref_material in mats_to_coref:
            alias_match = bool(material['alias'].lower() == coref_material['alias'].lower())
            #cde_record_match = any(bool(material['alias'] in rec and coref_material['alias'] in rec) for rec in cde_recs)
            #if alias_match or cde_record_match:
            if alias_match:
              modified_material = self._merge_materials(material, coref_material, merge_amounts=False)
        coreferenced_materials.append(modified_material)

      paper['materials'] = coreferenced_materials
    except Exception, e:
      self.__logger.warning( 'FAILURE: Failed to coreference target materials from ' + str(paper['doi']) )
      self.__logger.warning( 'ERR_MSG: ' + str(e) )

    self.connection[self.db].papers.update_one({'_id' : paper['_id']}, {'$set': paper})
    if save_verbose and 'recipe_text' in verbose_data: self.save_verbose_info(verbose_data)

  def apply_token_labels(self, tokens_text, cde_doc=None):
    #Run a chemdataextractor parse as well
    cde_cems = set([unicode(c) for c in cde_doc.cems])
    cde_abbrevs = []
    for abbrev_list in cde_doc.abbreviation_definitions:
      cde_abbrevs.extend(abbrev_list[0])

    #First, get some basic parse labels (SpaCy)
    spacy_tokens = self.nlp(tokens_text)
    spacy_noun_chunks = spacy_tokens.noun_chunks
    spacy_entities = spacy_tokens.ents
    toks_labels = []

    #Iterate through tokens and apply labels
    for i, tok in enumerate(spacy_tokens):
      tok_label = {}

      #Generic grammatical / language labels
      tok_label['spacy_tok'] = tok
      tok_label['head_raw'] = tok.head.text
      tok_label['head_lemma'] = tok.head.lemma_
      tok_label['raw'] = tok.text
      tok_label['raw_ws']  = tok.text_with_ws
      tok_label['lemma'] = tok.lemma_
      tok_label['ancestors'] = list(tok.ancestors)
      tok_label['raw_ancestors'] = ' '.join([t.text for t in tok.ancestors])
      tok_label['subtree'] = list(tok.subtree)
      tok_label['raw_subtree'] = ' '.join([t.text for t in tok.subtree])
      tok_label['like_num'] = tok.like_num
      tok_label['pos'] = tok.pos_
      tok_label['tag'] = tok.tag_
      tok_label['ent_type'] = tok.ent_type_
      tok_label['ent_iob'] = tok.ent_iob_
      tok_label['dep'] = tok.dep_
      tok_label['is_stop'] = tok.is_stop
      tok_label['head_pos'] = tok.head.pos_
      tok_label['head_tag'] = tok.head.tag_
      tok_label['property_type'] = self._list_match(tok_label['raw_ancestors'], self.property_types)
      tok_label['in_cde_cems'] = bool(tok_label['raw'] in cde_cems)
      tok_label['in_cde_abbrevs'] = bool(tok_label['raw'] in cde_abbrevs)

      #Position labels
      tok_label['tok_ind'] = i
      tok_label['start_char'] = tok.idx
      tok_label['end_char'] = tok.idx + len(tok_label['raw'])

      #Token filters
      tok_label['no_nums'] = sub('[0-9]', '', tok_label['raw'])
      tok_label['alphanum'] = sub('[^0-9a-zA-Z]+', '', tok_label['raw'])
      tok_label['alpha'] = sub('[^a-zA-Z]+', '', tok_label['raw'])
      tok_label['nums'] = sub('[^0-9]+', '', tok_label['raw'])

      #Spacy objects (to which this token belongs)
      tok_label['spacy_noun_chunk'] = ''
      for n_chk in spacy_noun_chunks:
        if (tok_label['start_char'] >= n_chk.start_char) and (tok_label['end_char'] <= n_chk.end_char):
          tok_label['spacy_noun_chunk'] = n_chk.text
          break
      tok_label['spacy_entity'] = ''
      for n_chk in spacy_entities:
        if (tok_label['start_char'] >= n_chk.start_char) and (tok_label['end_char'] <= n_chk.end_char):
          tok_label['spacy_entity'] = n_chk.text
          break

      tok_label['features'], tok_label['vec'] = self.tc.featurize(tok_label)
      toks_labels.append(tok_label)

    return toks_labels

  def get_relevant_chunks(self, labels):
    token_chunks = []
    curr_chunk = self._get_new_chunk()
    allowed_types = ['amount', 'condition', 'material', 'target', 'operation', 'descriptor', 'property', 'apparatus', 'intermediate', 'num', 'null']
    char_counter = 0

    label_vec_matrix = [
      np.array([l['vec'] for l in labels]),
      np.array([self.tc.featurize_embedding(l['raw']) for l in labels])
    ]
    label_types = [self.token_class_lookups[c] for c in self.tc.predict(label_vec_matrix)]

    for i, label in enumerate(labels):
      #print i, (label['raw'])
      curr_type_guess = label_types[i]

      if curr_type_guess == 'num': curr_type_guess = 'amount|condition|property'

      if ( (curr_type_guess not in curr_chunk['type'] and curr_chunk['type'] == 'null' and curr_type_guess != 'null')
      or (curr_type_guess in curr_chunk['type'] and curr_chunk['type'] != 'null' and curr_type_guess != 'null') ):  #Chunk start/inside
        curr_chunk['toks'].append(label)
        curr_chunk['char_len'] += len(label['raw_ws'])
        if curr_chunk['ind'] < 0: curr_chunk['ind'] = i
        curr_chunk['char_ind'] = char_counter
        curr_chunk['type'] = curr_type_guess
        #print '----START//CONTN: ', curr_type_guess

      if curr_chunk['type'] != 'null' and curr_type_guess == 'null': #Chunk end to None
        if curr_chunk['type'] in allowed_types:
          token_chunks.append(curr_chunk)
          #print '----CHUNK-->NONE: ', curr_type_guess
        curr_chunk = self._get_new_chunk()

      if curr_type_guess not in curr_chunk['type'] and curr_chunk['type'] != 'null' and curr_type_guess != 'null': #Chunk end to new chunk
        if curr_chunk['type'] in allowed_types:
          token_chunks.append(curr_chunk)
        curr_chunk = self._get_new_chunk()
        #print '----CHUNK->CHUNK: ', curr_type_guess
        curr_chunk['ind'] = i
        curr_chunk['char_ind'] = char_counter
        curr_chunk['type'] = curr_type_guess
        curr_chunk['toks'].append(label)
        curr_chunk['char_len'] += len(label['raw_ws'])

      char_counter += len(label['raw_ws'])

    return token_chunks, labels

  def get_recipe_from_chunks(self, chunks, assemble_connections=True):
    material_inds = []
    material_chunks = []
    materials = []
    operation_inds = []
    operation_chunks = []
    operations = []
    connections = []

    MAX_DIST = 9999

    # Get all materials and operations first
    try:
      for i, chunk in enumerate([c for c in chunks if c['type'] == 'operation']):
        new_operation = Operation()
        new_operation['_id'] = unicode(ObjectId())
        new_operation['order'] = i+1
        new_operation['type'] = chunk['toks'][0]['lemma']
        if any(t['features']['is_hydrothermal'] for t in chunk['toks']):
          new_operation['type'] = unicode('hydrothermal_') + unicode(new_operation['type'])
        operations.append(new_operation)
        operation_inds.append(chunk['ind'])
        operation_chunks.append(chunk)
    except Exception, e:
      self.__logger.warning( 'FAILURE: Failed to resolve operations')
      self.__logger.warning( 'ERR_MSG: ' + str(e) )

    try:
      if len(operation_inds) > 0:
        for i, chunk in enumerate([c for c in chunks if c['type'] in ['material', 'target', 'intermediate']]):
          if any(self._is_chunk_ancestor(chk, chunk) for chk in operation_chunks):
            new_material = Material()
            new_material['_id'] = unicode(ObjectId())
            new_material['is_target'] = bool(chunk['type'] == 'target')
            used_chunk = False
            for tok in chunk['toks']:
              if len(unicode(tok['spacy_noun_chunk'])) > 3:
                new_material['alias'] = unicode(tok['spacy_noun_chunk'])
                used_chunk = True
            if not used_chunk: new_material['alias'] = ' '.join(t['raw'] for t in chunk['toks'])
            if chunk['type'] == 'intermediate':
              new_material['alias'] = unicode('intermediate_') + unicode(new_material['alias'])

            materials.append(new_material)
            material_inds.append(chunk['ind'])
            material_chunks.append(chunk)
    except Exception, e:
      self.__logger.warning( 'FAILURE: Failed to resolve materials')
      self.__logger.warning( 'ERR_MSG: ' + str(e) )

    try:
      #Then sweep through all other chunks and get everything else (local mapping)
      for chunk in [c for c in chunks if c['type'] not in ['operation', 'material', 'target', 'intermediate']]:
        if len(material_inds) > 0:
          if chunk['type'] == 'amount':
            try:
              new_amount = Amount()
              num_conv = self._get_float_from_string(chunk['toks'][0]['raw'])
              if num_conv is None: num_conv = self._get_float_from_string(chunk['toks'][0]['nums'])
              if num_conv is None:
                new_amount['misc'] = ' '.join([t['raw'] for t in chunk['toks']])
                if len(new_amount['misc']) <= 3: continue
              else:
                new_amount['num_value'] = num_conv
                raw_units = ''.join([t['alpha'] for t in chunk['toks']])
                if len(raw_units) > 2: raw_units = raw_units.lower()

                if raw_units in self.unit_conv:
                  new_amount['num_value'] = (new_amount['num_value'] + self.unit_conv[raw_units][0]) * self.unit_conv[raw_units][1]
                  new_amount['units'] = unicode(self.unit_conv[raw_units][2])
                else:
                  new_amount['units'] = unicode(raw_units)

              chunk_dists = [abs(ind - chunk['ind']) if self._is_chunk_ancestor(chk, chunk) else MAX_DIST for (ind, chk) in zip(material_inds, material_chunks)]
              if not chunk_dists: chunk_dists = [abs(ind - chunk['ind']) for ind in material_inds]
              nearest_idx = chunk_dists.index(min(chunk_dists))
              materials[nearest_idx]['amounts'].append(new_amount)
            except Exception, e:
              self.__logger.warning( 'FAILURE: Failed amount casting')
              self.__logger.warning( 'ERR_MSG: ' + str(e) )

          if chunk['type'] == 'descriptor':
            try:
              new_descriptor = Descriptor()
              new_descriptor['structure'].append(' '.join(c['raw'] for c in chunk['toks']))

              chunk_dists = [abs(ind - chunk['ind']) if self._is_chunk_ancestor(chk, chunk) else MAX_DIST for (ind, chk) in zip(material_inds, material_chunks)]
              if not chunk_dists: chunk_dists = [abs(ind - chunk['ind']) for ind in material_inds]
              nearest_idx = chunk_dists.index(min(chunk_dists))
              materials[nearest_idx]['descriptors'].append(new_descriptor)
            except Exception, e:
              self.__logger.warning( 'FAILURE: Failed descriptor casting')
              self.__logger.warning( 'ERR_MSG: ' + str(e) )

          if chunk['type'] == 'property':
            try:
              new_property = Property()
              num_conv = self._get_float_from_string(chunk['toks'][0]['raw'])
              if num_conv is None: num_conv = self._get_float_from_string(chunk['toks'][0]['nums'])
              if num_conv is None: continue
              new_property['num_value'] = num_conv
              raw_units = ''.join([t['raw'] for t in chunk['toks']])
              new_property['units'] = unicode(raw_units)

              for tok in chunk['toks']:
                if tok['property_type'] is not None:
                  new_property['type'] = tok['property_type']
                  break

              chunk_dists = [abs(ind - chunk['ind']) if self._is_chunk_ancestor(chk, chunk) else MAX_DIST for (ind, chk) in zip(material_inds, material_chunks)]
              if not chunk_dists: chunk_dists = [abs(ind - chunk['ind']) for ind in material_inds]
              nearest_idx = chunk_dists.index(min(chunk_dists))
              materials[nearest_idx]['properties'].append(new_property)
            except Exception, e:
              self.__logger.warning( 'FAILURE: Failed property casting')
              self.__logger.warning( 'ERR_MSG: ' + str(e) )


        if len(operation_inds) > 0:
          if chunk['type'] == 'condition':
            try:
              new_condition = Condition()
              num_conv = self._get_float_from_string(chunk['toks'][0]['raw'])
              if num_conv is None: num_conv = self._get_float_from_string(chunk['toks'][0]['nums'])
              if num_conv is None:
                new_condition['misc'] = ' '.join([t['raw'] for t in chunk['toks']])
                if len(new_condition['misc']) <= 3: continue
              else:
                new_condition['num_value'] = num_conv
                raw_units = ''.join([t['alpha'] for t in chunk['toks']])
                if len(raw_units) > 2: raw_units = raw_units.lower()

                if raw_units in self.unit_conv:
                  new_condition['num_value'] = (new_condition['num_value'] + self.unit_conv[raw_units][0]) * self.unit_conv[raw_units][1]
                  new_condition['units'] = unicode(self.unit_conv[raw_units][2])
                else:
                  if any(t['features']['is_ph_val'] for t in chunk['toks']):
                    new_condition['units'] = unicode('pH')
                  else:
                    new_condition['units'] = unicode(raw_units)

              chunk_dists = [abs(ind - chunk['ind']) if self._is_chunk_ancestor(chk, chunk) else MAX_DIST for (ind, chk) in zip(operation_inds, operation_chunks)]
              if not chunk_dists: chunk_dists = [abs(ind - chunk['ind']) for ind in operation_inds]
              nearest_idx = chunk_dists.index(min(chunk_dists))
              operations[nearest_idx]['conditions'].append(new_condition)
            except Exception, e:
              self.__logger.warning( 'FAILURE: Failed condition casting')
              self.__logger.warning( 'ERR_MSG: ' + str(e) )

          if chunk['type'] == 'apparatus':
            try:
              new_apparatus = Apparatus()
              new_apparatus['type'] = ' '.join(c['raw'] for c in chunk['toks'])

              chunk_dists = [abs(ind - chunk['ind']) if self._is_chunk_ancestor(chk, chunk) else MAX_DIST for (ind, chk) in zip(operation_inds, operation_chunks)]
              if not chunk_dists: chunk_dists = [abs(ind - chunk['ind']) for ind in operation_inds]
              nearest_idx = chunk_dists.index(min(chunk_dists))
              operations[nearest_idx]['apparatuses'].append(new_apparatus)
            except Exception, e:
              self.__logger.warning( 'FAILURE: Failed apparatus casting')
              self.__logger.warning( 'ERR_MSG: ' + str(e) )
    except Exception, e:
      self.__logger.warning( 'FAILURE: Failed to resolve local mappings' )
      self.__logger.warning( 'ERR_MSG: ' + str(e) )

    if assemble_connections:
      try:
        #Finally, put together some connections
        used_material_ids = set()
        for ind, operation in enumerate(operations):
          input_materials = [m for i, m in zip(material_inds, materials) if i <= operation_inds[ind] and not m['is_target'] and m['_id'] not in used_material_ids]
          op_id = operation['_id']
          input_ids = [m['_id'] for m in input_materials if m['_id'] not in used_material_ids]
          used_material_ids.update(input_ids) #Prevents double counting of inputs

          #Create dummy intermediates as needed
          need_intermediate = bool(not any('intermediate' in m['alias'] for m in input_materials))
          if ind > 0 and need_intermediate:
            intermediate_id = 'placeholder_' + unicode(ObjectId())
            input_ids.append(intermediate_id)

          #Link up inputs to operation
          for input_id in input_ids:
            new_connection = Connection()
            new_connection['id1'] = input_id
            new_connection['id2'] = op_id
            connections.append(new_connection)

          #Link up outputs from previous operation
          if ind > 0:
            prev_op_id = operations[ind-1]['_id']
            new_connection = Connection()
            new_connection['id1'] = prev_op_id

            if need_intermediate:
              new_connection['id2'] = intermediate_id
            else:
              new_connection['id2'] = next(m['_id'] for m in input_materials if 'intermediate' in m['alias'])

            connections.append(new_connection)

        #Add in the final outputs
        if len(operations) > 1:
          final_output_ids = [m['_id'] for m in materials if m['_id'] not in used_material_ids]
          for output_id in final_output_ids:
            new_connection = Connection()
            new_connection['id1'] = operations[-1]['_id']
            new_connection['id2'] = output_id
            connections.append(new_connection)
      except Exception, e:
        self.__logger.warning( 'FAILURE: Failed to resolve connections: ' + str(traceback.format_exc()) )
        self.__logger.warning( 'ERR_MSG: ' + str(e) )


    return materials, operations, connections

  def refine_recipe(self, materials, operations, connections, chunks):
    #First, look for operations that seem extraneous or don't actually do anything
    #Also do a fix for hydrothermal route detection
    del_indices = []
    is_route_hydrothermal = False
    for i, operation in enumerate(operations):
      if 'hydrothermal_' in operation['type']: is_route_hydrothermal = True
      if is_route_hydrothermal and 'hydrothermal_' not in operation['type']:
          if operation['type'] in self.operation_categories['heat'] + self.operation_categories['misc']:
              if operation['type'] not in ['calcine', 'anneal', 'sinter', 'fire', 'pyrolyze', 'melt']:
                  operation['type'] = 'hydrothermal_' + operation['type']

      #Prune operations with no inputs
      inputs = [c for c in connections if c['id2'] == operation['_id']]
      if not inputs: del_indices.append(i)

    new_operations = [o for i, o in enumerate(operations) if i not in del_indices]
    for i, operation in enumerate(new_operations): operation['order'] = i+1
    new_connections = [c for c in connections if c['id1'] not in [del_indices]]

    #Prune placeholder intermediates left 'hanging'
    del_indices = []
    for i, connection in enumerate(new_connections):
      id1 = connection['id1']
      id2 = connection['id2']
      if 'placeholder' in id2 and id1 not in [o['_id'] for o in new_operations]:
        del_indices.append(id2)

    new_connections = [c for c in new_connections if c['id1'] not in del_indices and c['id2'] not in del_indices]

    #Prune duplicate materials
    duplicate_sets = {}
    for material in materials:
      alias = material['alias']
      common_in = next((c['id1'] for c in new_connections if c['id2'] == material['_id']), None)
      common_out = next((c['id2'] for c in new_connections if c['id1'] == material['_id']), None)
      name_matches = [m for m in materials if m['alias'] == alias]
      id_matches = []

      for name_match in name_matches:
        this_in = next((c['id1'] for c in new_connections if c['id2'] == name_match['_id']), None)
        this_out = next((c['id2'] for c in new_connections if c['id1'] == name_match['_id']), None)
        if this_in == common_in and this_out == common_out:
          id_matches.append(name_match['_id'])

      if len(id_matches) > 1 and alias not in duplicate_sets:
        duplicate_sets[alias] = (id_matches, common_in, common_out)

    del_indices = []
    new_materials = []
    for dup_key in duplicate_sets:
      mat_ids = duplicate_sets[dup_key][0]
      mat_to_keep = next(m for m in materials if m['_id'] == mat_ids[0])
      del_indices.append(mat_to_keep)

      if len(mat_ids) > 1:
        for mat_id in mat_ids[1:]:
          other_mat = next(m for m in materials if m['_id'] == mat_id)
          mat_to_keep = self._merge_materials(mat_to_keep, other_mat)
          del_indices.append(mat_id)

      new_materials.append(mat_to_keep)

    new_materials.extend([m for m in materials if m['_id'] not in del_indices])
    new_connections = [c for c in new_connections if c['id1'] not in del_indices and c['id2'] not in del_indices]

    return new_materials, new_operations, new_connections

  def _load_token_classifier(self):
    self.tc = TokenClassifier(self.db)
    self.tc.load()

  def _get_new_chunk(self):
    chunk = {
      'ind': -1,
      'char_ind': -1,
      'char_len': 0,
      'type': '',
      'toks': []
    }
    return chunk

  def _is_chunk_ancestor(self, parent_chunk, child_chunk):
    p_toks = [t['spacy_tok'] for t in parent_chunk['toks']]
    c_toks = [t['spacy_tok'] for t in child_chunk['toks']]

    return any(p.is_ancestor(c) for p in p_toks for c in c_toks)

  def _get_float_from_string(self, string):
    num = None
    try: num = float(string)
    except: pass
    if num is None:
      try: num = self.num_words[string.lower()]
      except: pass
    if num is None:
      try: num = float(sum(findall('\d+', string))) / len(string)
      except: pass
      if num == 0: return None
    return num

  def _merge_materials(self, mat_to_keep, mat, merge_amounts=True):
    mat_to_keep['properties'].extend(mat['properties'])
    mat_to_keep['descriptors'].extend(mat['descriptors'])
    if merge_amounts: mat_to_keep['amounts'].extend(mat['amounts'])

    return mat_to_keep

  def _list_match(self, search_string, potential_matches):
    for pot_match in potential_matches:
      if pot_match in search_string:
        return pot_match
    return None

  def _load_lexicon(self, lexicon_path='data/extraction_lexicon.json'):
    with open(lexicon_path) as f:
      self.lexicon = loads(f.read())

    #Properties
    self.property_types = self.lexicon['material_properties']

    #Known operations to match
    self.operation_categories = self.lexicon['operation_categories']
    self.operations = [] #Build flat list from category list
    self.operations_categories = set() #Build tuple set from category list
    for category in self.operation_categories:
      for operation in self.operation_categories[category]:
        self.operations.append(operation)
        self.operations_categories.add((category, operation))

  def _load_unit_conversions(self, fpath='data/unit_conversions.json'):
    with open(fpath) as f:
      self.unit_conv = loads(f.read())
