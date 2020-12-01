``pystiche_papers.li_wand_2016``
================================

+--------------------------+-----------------------------------------------------------+
| Title                    | Combining Markov Random Fields and Convolutional Neural   |
|                          |                                                           |
|                          | Networks for Image Synthesis                              |
+--------------------------+-----------------------------------------------------------+
| Authors                  | Chuan Li and Michael Wand                                 |
+--------------------------+-----------------------------------------------------------+
| Citation                 | :cite:`LW2016`                                            |
+--------------------------+-----------------------------------------------------------+
| Reference implementation | |repo|_ / |archive|_                                      |
+--------------------------+-----------------------------------------------------------+
| Variant                  | Image optimization                                        |
+--------------------------+-----------------------------------------------------------+
| Content loss             | :class:`~pystiche.ops.FeatureReconstructionOperator`      |
+--------------------------+-----------------------------------------------------------+
| Style loss               | :class:`~pystiche.ops.MRFOperator`                        |
+--------------------------+-----------------------------------------------------------+
| Regularization           | :class:`~pystiche.ops.TotalVariationOperator`             |
+--------------------------+-----------------------------------------------------------+

.. |repo| replace:: Repository
.. _repo: https://github.com/chuanli11/CNNMRF/

.. |archive| replace:: Archive
.. _archive: https://github.com/pmeier/CNNMRF/tree/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f

Unfortunately, the parameters in the reference implementation differ from the parameters
described in the paper. If ``impl_params is True``, the parameters from the reference
implementation are used instead of the parameters from the paper. The following
parts are affected:

  - :func:`~pystiche_papers.li_wand_2016.image_pyramid`,
  - :func:`~pystiche_papers.li_wand_2016.content_loss`,
  - :func:`~pystiche_papers.li_wand_2016.style_loss`,
  - :func:`~pystiche_papers.li_wand_2016.regularization`,
  - :func:`~pystiche_papers.li_wand_2016.target_transforms`.


.. automodule:: pystiche_papers.li_wand_2016

.. autofunction:: content_loss
.. autofunction:: extract_normalized_patches2d
.. autofunction:: images
.. autofunction:: image_pyramid
.. autofunction:: multi_layer_encoder
.. autofunction:: nst
.. autofunction:: optimizer
.. autofunction:: perceptual_loss
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: regularization
.. autofunction:: style_loss
.. autofunction:: target_transforms
