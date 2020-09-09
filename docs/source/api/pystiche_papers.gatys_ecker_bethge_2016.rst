``pystiche_papers.gatys_ecker_bethge_2016``
===========================================

+--------------------------+-----------------------------------------------------------+
| Title                    | Image Style Transfer Using Convolutional Neural Networks  |
+--------------------------+-----------------------------------------------------------+
| Authors                  | Leon A. Gatys, Alexander. S. Ecker, and Matthias Bethge   |
+--------------------------+-----------------------------------------------------------+
| Citation                 | :cite:`GEB2016`                                           |
+--------------------------+-----------------------------------------------------------+
| Reference implementation | |repo|_ / |archive|_                                      |
+--------------------------+-----------------------------------------------------------+
| Variant                  | Image optimization                                        |
+--------------------------+-----------------------------------------------------------+
| Content loss             | :class:`~pystiche.ops.FeatureReconstructionOperator`      |
+--------------------------+-----------------------------------------------------------+
| Style loss               | :class:`~pystiche.ops.GramOperator`                       |
+--------------------------+-----------------------------------------------------------+

.. |repo| replace:: Repository
.. _repo: https://github.com/leongatys/PytorchNeuralStyleTransfer

.. |archive| replace:: Archive
.. _archive: https://github.com/pmeier/PytorchNeuralStyleTransfer/tree/c673ff2dad4cebaf753aa94bf1658292d967058a

Unfortunately, the parameters in the reference implementation differ from the parameters
described in the paper. If ``impl_params is True``, the parameters from the reference
implementation are used instead of the parameters from the paper. The following parts
are affected:

  - :func:`~pystiche_papers.gatys_ecker_bethge_2016.nst`,
  - :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder`,
  - :func:`~pystiche_papers.gatys_ecker_bethge_2016.content_loss`,
  - :func:`~pystiche_papers.gatys_ecker_bethge_2016.style_loss`.

.. automodule:: pystiche_papers.gatys_ecker_bethge_2016

.. autofunction:: content_loss
.. autofunction:: images
.. autofunction:: multi_layer_encoder
.. autofunction:: nst
.. autofunction:: optimizer
.. autofunction:: perceptual_loss
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: style_loss
