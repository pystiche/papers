``pystiche_papers.ulyanov_et_al_2016``
=========================================

Unfortunately, the parameters in the reference implementation differ from the parameters
described in the paper. If ``impl_params is True``, the parameters from the reference
implementation are used instead of the parameters from the paper. The following
functions are affected by the parameter ``impl_params``:

  - :func:`~pystiche_papers.ulyanov_et_al_2016.training`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.stylization`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.optimizer`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.lr_scheduler`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.content_transform`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.style_transform`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.batch_sampler`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.content_loss`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.style_loss`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.transformer`.



.. automodule:: pystiche_papers.ulyanov_et_al_2016

.. autofunction:: batch_sampler
.. autofunction:: content_loss
.. autofunction:: content_transform
.. autofunction:: dataset
.. autofunction:: images
.. autofunction:: image_loader
.. autofunction:: multi_layer_encoder
.. autofunction:: optimizer
.. autofunction:: perceptual_loss
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: style_loss
.. autofunction:: style_transform
.. autofunction:: stylization
.. autofunction:: training
.. autofunction:: transformer
