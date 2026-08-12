"""
models package
----------------
Exposes the SVM classes so other scripts can do:
    from models import OvRSVM, KernelOvRSVM
instead of importing from individual files.
"""

from svm import BinarySVM, OvRSVM
from kernel import polynomial_kernel, rbf_kernel, KernelBinarySVM, KernelOvRSVM
from confidence import softmax_confidence, predict_with_confidence, calibration_check

__all__ = [
    "BinarySVM", "OvRSVM",
    "polynomial_kernel", "rbf_kernel", "KernelBinarySVM", "KernelOvRSVM",
    "softmax_confidence", "predict_with_confidence", "calibration_check",
]