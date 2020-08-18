``pystiche_papers.li_wand_2016``
================================

Unfortunately, the parameters in the reference implementation differ from the parameters
described in the paper. If ``impl_params is True``, the parameters from the reference
implementation are used instead of the parameters from the paper. The following
parts are affected:

  - :func:`~pystiche_papers.li_wand_2016.image_pyramid`,
  - :func:`~pystiche_papers.li_wand_2016.content_loss`,
  - :func:`~pystiche_papers.li_wand_2016.style_loss`,
  - :func:`~pystiche_papers.li_wand_2016.regularization`.



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
