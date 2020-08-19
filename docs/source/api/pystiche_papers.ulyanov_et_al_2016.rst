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


.. _table-branches-ulyanov_et_al_2016:

The original authors have provided two implementations. Unfortunately, the
hyperparameters used differ from those in the paper, as well as
among the individual implementations themselves. If you use ``instance_norm`` and
``impl_params`` as in the following table, the appropriate parameters will be used:

  - ``master``: The parameters specified in the implementation branch
    `master <https://github.com/pmeier/texture_nets/tree/master>`_.
  - ``"texture_nets_v1"``: The parameters specified in the implementation branch
    `texture_nets_v1 <https://github.com/pmeier/texture_nets/tree/texture_nets_v1>`_.
  - ``"paper"``: The parameters specified in the paper. Here instance_norm is only used
    to use either :class:`~torch.nn.InstanceNorm2d` or :class:`~torch.nn.BatchNorm2d`.

+-----------------+-----------+------------------------------------+
|                 |           |        ``instance_norm``           |
+-----------------+-----------+-------------+----------------------+
|                 |           |   ``True``  |       ``False``      |
+-----------------+-----------+-------------+----------------------+
| ``impl_params`` | ``True``  | ``"master"``|``"texture_nets_v1"`` |
|                 +-----------+-------------+----------------------+
|                 | ``False`` | ``"paper"`` |      ``"paper"``     |
+-----------------+-----------+-------------+----------------------+

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

The ``layers`` have either ``4 layers`` ``("relu1_1", "relu2_1", "relu3_1", "relu4_1")``
or ``5 Layer`` ``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")``.


+-------------------+---------------+--------------+-----------------------+
|   Parameter       |  ``"paper"``  | ``"master"`` | ``"texture_nets_v1"`` |
+===================+===============+==============+=======================+
|  ``num_epochs``   |      10       |      10      |           25          |
+-------------------+---------------+--------------+-----------------------+
|     ``lr``        |     1e-1      |    1e-3      |         1e-1          |
+-------------------+---------------+--------------+-----------------------+
|  ``num_batches``  |     200       |     2000     |          300          |
+-------------------+---------------+--------------+-----------------------+
|  ``batch_size``   |      16       |       1      |           4           |
+-------------------+---------------+--------------+-----------------------+
| ``content_loss``  |               |              |                       |
+-------------------+---------------+--------------+-----------------------+
| ``score_weight``  |      1e0      |      1e0     |          6e1          |
+-------------------+---------------+--------------+-----------------------+
| ``style_loss``    |               |              |                       |
+-------------------+---------------+--------------+-----------------------+
| ``score_weight``  |      1e0      |      1e0     |          1e3          |
+-------------------+---------------+--------------+-----------------------+
| ``layers``        |  ``5 Layer``  |  ``4 Layer`` |     ``5 Layer``       |
+-------------------+---------------+--------------+-----------------------+


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
