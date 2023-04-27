# pylint: disable=wrong-import-position,wrong-import-order

from types import ModuleType
from typing import Union

try:
    from importlib import metadata
except (ModuleNotFoundError, ImportError):
    # Python <= 3.7
    import importlib_metadata as metadata  # type: ignore

try:
    version = metadata.version("farm_haystack")
except metadata.PackageNotFoundError:
    version = "0.0.0"  # Specify a default version if metadata is not found

__version__: str = str(version)


# Logging is not configured here on purpose, see https://github.com/deepset-ai/haystack/issues/2485
import logging

import pandas as pd

from haystack.environment import set_pytorch_secure_model_loading
from haystack.nodes.base import BaseComponent
from haystack.pipelines.base import Pipeline
from haystack.schema import Answer, Document, EvaluationResult, Label, MultiLabel, Span, TableCell

pd.options.display.max_colwidth = 80
set_pytorch_secure_model_loading()
