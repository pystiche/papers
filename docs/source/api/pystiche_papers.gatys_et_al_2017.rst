``pystiche_papers.gatys_et_al_2017``
====================================

Unfortunately, the parameters in the reference implementation differ from the parameters
described in the paper. If ``impl_params is True``, the parameters from the reference
implementation are used instead of the parameters from the paper. The following parts
are affected:

  - :func:`~pystiche_papers.gatys_ecker_bethge_2016.style_loss`,
  - :func:`~pystiche_papers.gatys_ecker_bethge_2016.guided_style_loss`.

.. automodule:: pystiche_papers.gatys_et_al_2017

.. autofunction:: content_loss
.. autofunction:: guided_nst
.. autofunction:: guided_perceptual_loss
.. autofunction:: guided_style_loss
.. autofunction:: images
.. autofunction:: image_pyramid
.. autofunction:: multi_layer_encoder
.. autofunction:: nst
.. autofunction:: optimizer
.. autofunction:: perceptual_loss
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: style_loss
