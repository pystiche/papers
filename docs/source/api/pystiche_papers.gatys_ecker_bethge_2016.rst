``pystiche_papers.gatys_ecker_bethge_2015``
===========================================

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
