"""Canonical title keys for study-selection joins."""

from src.data.selection.title_key import canonical_title_key

_MULAYER_MICRO = (
    "µLayer: Low latency on-device inference using cooperative "
    "single-layer acceleration and processor-friendly quantization"
)
_MULAYER_MU = (
    "μLayer: Low latency on-device inference using cooperative "
    "single-layer acceleration and processor-friendly quantization"
)
_OBJECT_DETECTION = "Quantization and training of low bit-width convolutional neural networks for object detection*"
_MIX_GEMM_USING = (
    "Mix-GEMM: Extending RISC-V CPUs for Energy-Efficient Mixed-Precision DNN Inference using Binary Segmentation"
)
_MIX_GEMM_USING_CAP = (
    "Mix-GEMM: Extending RISC-V CPUs for Energy-Efficient Mixed-Precision DNN Inference Using Binary Segmentation"
)


def test_canonical_title_key_unifies_micro_sign_and_case():
    assert canonical_title_key(_MULAYER_MICRO) == canonical_title_key(_MULAYER_MU)


def test_canonical_title_key_collapses_whitespace_and_trailing_asterisk():
    messy = _OBJECT_DETECTION + "                          "
    assert canonical_title_key(messy) == canonical_title_key(_OBJECT_DETECTION)


def test_canonical_title_key_casefolds_using():
    assert canonical_title_key(_MIX_GEMM_USING) == canonical_title_key(_MIX_GEMM_USING_CAP)
