from typing import Optional

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche.enc.encoder import SequentialEncoder
from pystiche.ops.comparison import FeatureReconstructionOperator

__all__ = ["transformed_image_loss"]


def transformed_image_loss(
    transformer_block: Optional[SequentialEncoder] = None,
    impl_params: bool = True,
    score_weight: Optional[float] = None,
) -> FeatureReconstructionOperator:
    r"""Transformed_image_loss from from :cite:`SKL+2018`.

    Args:
        transformer_block::class:`~pystiche_papers.sanakoyeu_et_al_2018.TransformerBlock`
            which is used to transform the image.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        score_weight: Score weight of the operator. If omitted, the score_weight is
            determined with respect to ``impl_params``. Defaults to ``1e2`` if
            ``impl_params is True`` otherwise ``1e0``.
    """
    if score_weight is None:
        score_weight = 1e2 if impl_params else 1e0

    if transformer_block is None:
        transformer_block = paper.TransformerBlock()

    return FeatureReconstructionOperator(transformer_block, score_weight=score_weight)
