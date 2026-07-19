"""Autodata: dataset-level synthetic data construction."""

from .models import Candidate, DatasetReport, TaskSpec
from .meta import MetaOptimizer
from .discovery import ContractSynthesizer, SourceProfiler
from .utility import AdaptiveUtilityGate, PilotResult
from .pipeline import DatasetBuilder
from .context import ContextController, ContextOverflow, ContextPlan, ContextPolicy, ContextSegment

__all__ = ["AdaptiveUtilityGate", "Candidate", "ContextController", "ContextOverflow", "ContextPlan", "ContextPolicy", "ContextSegment", "ContractSynthesizer", "DatasetBuilder", "DatasetReport", "MetaOptimizer", "PilotResult", "SourceProfiler", "TaskSpec"]
