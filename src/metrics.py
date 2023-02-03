import sklearn.metrics
import numpy as np
from scipy.stats import pearsonr as scipy_pearsonr
import re
from datasets import load_metric

def map_name_to_metric_function(name):
  dict_ = {
    "rouge": rouge,
  }
  return dict_[name]


def rouge(targets, predictions):
  rouge = load_metric('rouge')
  results = rouge.compute(predictions=predictions,references=targets)
  results = {k:v.mid.fmeasure for k,v in results.items()}
  return results

def accuracy(targets, predictions):
  return {"accuracy": 100*sklearn.metrics.accuracy_score(targets, predictions)}

"""Functions for computing metrics.
Every function must accept a list of targets and a list of predictions and
return a dict of metrics.
Functions should assume all text inputs are unicode strings.
"""
def f1(targets, predictions, average="micro"):
  return {f"f1_{average}": 100*sklearn.metrics.f1_score(targets, predictions, average=average)}



