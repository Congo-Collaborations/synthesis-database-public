import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from synthesisdatabase.models import (Paragraph, Paper, Material, Connection, Operation)
from synthesisdatabase.classifiers.synth_para_classifier import (SynthParaClassifier)
from synthesisdatabase.classifiers.token_classifier import (TokenClassifier)
from synthesisdatabase.classifiers.material_token_classifier import (MaterialTokenClassifier)
from synthesisdatabase.extractors.info_extractor import (InfoExtractor)
from synthesisdatabase.managers.download_manager import (DownloadManager)

from unittest import TestCase


class ModelTester(TestCase):
  def setUp(self):
    self.paper = Paper()
    self.material = Material()
    self.operation = Operation()

  def test_build(self):
    self.assertTrue(self.paper is not None)
    self.assertTrue(self.material is not None)
    self.assertTrue(self.operation is not None)

class ClassifiersTester(TestCase):
  def setUp(self):
    self.db = 'predsynth'
    self.spc = SynthParaClassifier(self.db)
    self.tc = TokenClassifier(self.db)
    self.mc = MaterialTokenClassifier()

  def test_build(self):
    self.tc.build_nn_model()

  def test_load(self):
    self.tc._load_compound_names()
    self.tc._load_lexicon()

  def test_train(self):
    self.mc.train()

  def test_thres(self):
    self.assertTrue(self.spc.set_threshold(0))
    self.assertTrue(self.spc.set_threshold(0.5))
    self.assertFalse(self.spc.set_threshold(-1))

class ExtractorTester(TestCase):
  def setUp(self):
    self.db = 'predsynth'
    self.ie = InfoExtractor(self.db)

  def test_build(self):
    self.assertTrue(self.ie is not None)

  def test_load(self):
    self.ie.load_all_papers(0)

class ManagerTester(TestCase):
  def setUp(self):
    self.db = 'predsynth'
    self.dm = DownloadManager(self.db)

  def test_build(self):
    self.assertTrue(self.dm is not None)

  def test_relevance(self):
    self.assertTrue(self.dm.is_title_relevant('Synthesis of TiO2 nanotubes'))
    self.assertFalse(self.dm.is_title_relevant('DNA experiments with bacterial cells'))
