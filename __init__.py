from .pag_nodes import (
    PerturbedAttention,
    MultiBlockPerturbedAttention,
    SmoothedEnergyGuidanceAdvanced,
    SlidingWindowGuidanceAdvanced,
)
from .pag_trt_nodes import TRTAttachPag, TRTPerturbedAttention

NODE_CLASS_MAPPINGS = {
    "PerturbedAttention": PerturbedAttention,
    "MultiBlockPerturbedAttention": MultiBlockPerturbedAttention,
    "SmoothedEnergyGuidanceAdvanced": SmoothedEnergyGuidanceAdvanced,
    "SlidingWindowGuidanceAdvanced": SlidingWindowGuidanceAdvanced,
    "TRTAttachPag": TRTAttachPag,
    "TRTPerturbedAttention": TRTPerturbedAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerturbedAttention": "The Perturbinator",
    "MultiBlockPerturbedAttention": "Multi-Block Perturbinator",
    "SmoothedEnergyGuidanceAdvanced": "Smoothed Energy Guidance (Advanced)",
    "SlidingWindowGuidanceAdvanced": "Sliding Window Guidance (Advanced)",
    "TRTAttachPag": "TensorRT Attach PAG",
    "TRTPerturbedAttention": "TensorRT Perturbed-Attention Guidance",
}
