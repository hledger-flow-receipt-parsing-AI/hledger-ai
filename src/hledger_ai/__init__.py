"""hledger-ai: Local-first AI models for receipt OCR and transaction classification."""

import os
import warnings

# Suppress TensorFlow/CUDA/absl/transformers warnings early
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_DEPRECATION_WARNINGS", "0")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")

__version__ = "0.1.0"
