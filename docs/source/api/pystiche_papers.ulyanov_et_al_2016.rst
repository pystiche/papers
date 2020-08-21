``pystiche_papers.ulyanov_et_al_2016``
=========================================

Unfortunately, the parameters in the reference implementation differ from the parameters
described in the paper. If ``impl_params is True``, the parameters from the reference
implementation are used instead of the parameters from the paper. The following parts
are affected:

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


.. _table-branches-ulyanov_et_al_2016:

The original authors experimented with the network architecture and described two
versions of the network in different papers, replacing the
:class:`~torch.nn.BatchNorm2d` by :class:`~torch.nn.InstanceNorm2d` as an improvement.
They call it StyleNet with a postfix BN if :class:`~torch.nn.BatchNorm2d` or IN if
:class:`~torch.nn.InstanceNorm2d` is used:

  - ``master``: Corresponds to the reference implementation of the
    `StyleNet_IN <https://arxiv.org/abs/1701.02096>`_.
  - ``texture_nets_v1``: Corresponds to the reference implementation of the
    `StyleNet_BN <https://arxiv.org/abs/1603.03417>`_.

Unfortunately, the hyperparameters used differ from those in the papers, as well as
among the individual implementations themselves. If you use ``instance_norm`` and
``impl_params``, the appropriate parameters will be used:

  - ``master``: The parameters specified in the implementation branch
    `master <https://github.com/pmeier/texture_nets/tree/master>`_.
  - ``texture_nets_v1``: The parameters specified in the implementation branch
    `texture_nets_v1 <https://github.com/pmeier/texture_nets/tree/texture_nets_v1>`_.

.. _table-hyperparameters-ulyanov_et_al_2016:

The following table provides an overview of the parameters:

- the ``num_epochs`` of the :func:`~pystiche_papers.ulyanov_et_al_2016.training`,

- the ``lr`` the learning rate of the
  :func:`~pystiche_papers.ulyanov_et_al_2016.optimizer`,

- the ``num_batches`` and ``batch_size`` of the
  :func:`~pystiche_papers.ulyanov_et_al_2016.image_loader`,

- the ``score_weight`` for

  - the :func:`~pystiche_papers.ulyanov_et_al_2016.content_loss`,
  - the :func:`~pystiche_papers.ulyanov_et_al_2016.style_loss`, as well as

- the ``layers`` for

  - the :func:`~pystiche_papers.ulyanov_et_al_2016.style_loss`.

The ``layers`` have either configuration ``1``
``("relu1_1", "relu2_1", "relu3_1", "relu4_1")`` or  configuration ``2``
``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")``.


+---------------------------+-------------+---------------------+
| Parameter                 | ``master``  |``texture_nets_v1``  |
+===========================+=============+=====================+
| ``num_epochs``            | ``10``      | ``25``              |
+---------------------------+-------------+---------------------+
| ``learning_rate``         | ``1e-3``    | ``1e-1``            |
+---------------------------+-------------+---------------------+
| ``num_batches``           | ``2000``    | ``300``             |
+---------------------------+-------------+---------------------+
| ``batch_size``            | ``1``       | ``4``               |
+---------------------------+-------------+---------------------+
| ``content_score_weight``  | ``1e0``     | ``6e1``             |
+---------------------------+-------------+---------------------+
| ``style_score_weight``    | ``1e0``     | ``1e3``             |
+---------------------------+-------------+---------------------+
| ``style_layers``          | ``1``       | ``2``               |
+---------------------------+-------------+---------------------+


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
