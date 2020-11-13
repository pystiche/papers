``pystiche_papers.johnson_alahi_li_2016``
=========================================

+--------------------------+-----------------------------------------------------------+
| Title                    | Perceptual Losses for Real-Time Style Transfer and        |
|                          |                                                           |
|                          | Super-Resolution                                          |
+--------------------------+-----------------------------------------------------------+
| Authors                  | Justin Johnson, Alexandre Alahi, and Fei-Fei Li           |
+--------------------------+-----------------------------------------------------------+
| Citation                 | :cite:`JAL2016`                                           |
+--------------------------+-----------------------------------------------------------+
| Reference implementation | |repo|_ / |archive|_                                      |
+--------------------------+-----------------------------------------------------------+
| Variant                  | Model optimization                                        |
+--------------------------+-----------------------------------------------------------+
| Content loss             | :class:`~pystiche.ops.FeatureReconstructionOperator`      |
+--------------------------+-----------------------------------------------------------+
| Style loss               | :class:`~pystiche.ops.GramOperator`                       |
+--------------------------+-----------------------------------------------------------+
| Regularization           | :class:`~pystiche.ops.TotalVariationOperator`             |
+--------------------------+-----------------------------------------------------------+

.. |repo| replace:: Repository
.. _repo: https://github.com/jcjohnson/fast-neural-style

.. |archive| replace:: Archive
.. _archive: https://github.com/pmeier/fast-neural-style/tree/813c83441953ead2adb3f65f4cc2d5599d735fa7

Unfortunately, the parameters in the reference implementation differ from the parameters
described in the paper. If ``impl_params is True``, the parameters from the reference
implementation are used instead of the parameters from the paper. The following
parts are affected:

  - :func:`~pystiche_papers.johnson_alahi_li_2016.training`
  - :func:`~pystiche_papers.johnson_alahi_li_2016.stylization`
  - :func:`~pystiche_papers.johnson_alahi_li_2016.multi_layer_encoder`
  - :func:`~pystiche_papers.johnson_alahi_li_2016.content_transform`
  - :func:`~pystiche_papers.johnson_alahi_li_2016.style_transform`
  - :func:`~pystiche_papers.johnson_alahi_li_2016.batch_sampler`
  - :func:`~pystiche_papers.johnson_alahi_li_2016.transformer`
  - :func:`~pystiche_papers.johnson_alahi_li_2016.content_loss`
  - :func:`~pystiche_papers.johnson_alahi_li_2016.style_loss`
  - :func:`~pystiche_papers.johnson_alahi_li_2016.regularization`


.. automodule:: pystiche_papers.johnson_alahi_li_2016

.. autofunction:: batch_sampler
.. autofunction:: content_loss
.. autofunction:: content_transform
.. autofunction:: dataset
.. autofunction:: hyper_parameters
.. autofunction:: images
.. autofunction:: image_loader
.. autofunction:: multi_layer_encoder
.. autofunction:: optimizer
.. autofunction:: perceptual_loss
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: regularization
.. autofunction:: style_loss
.. autofunction:: style_transform
.. autofunction:: stylization
.. autofunction:: training
.. autofunction:: transformer
