``pystiche_papers.ulyanov_et_al_2016``
======================================

+--------------------------+-----------------------------------------------------------+
| Title                    | Texture Networks: Feed-forward Synthesis of Textures and  |
|                          |                                                           |
|                          | Stylized Images                                           |
+--------------------------+-----------------------------------------------------------+
| Authors                  | Dmitry Ulyanov, Vadim Lebedev, Andrea Vedaldi, and        |
|                          |                                                           |
|                          | Viktor S. Lempitsky                                       |
+--------------------------+-----------------------------------------------------------+
| Citation                 | :cite:`ULVL2016` / :cite:`UVL2017`                        |
+--------------------------+-----------------------------------------------------------+
| Reference implementation | |repo|_ / |archive|_                                      |
+--------------------------+-----------------------------------------------------------+
| Variant                  | Model optimization                                        |
+--------------------------+-----------------------------------------------------------+
| Content loss             | :class:`~pystiche.ops.FeatureReconstructionLoss`          |
+--------------------------+-----------------------------------------------------------+
| Style loss               | :class:`~pystiche.ops.GramLoss`                           |
+--------------------------+-----------------------------------------------------------+

.. |repo| replace:: Repository
.. _repo: https://github.com/DmitryUlyanov/texture_nets

.. |archive| replace:: Archive
.. _archive: https://github.com/pmeier/texture_nets

.. _ulyanov_et_al_2016-instance_norm:

Instance norm
-------------

The authors published an improved version :cite:`UVL2017` of their initial paper
:cite:`ULVL2016` with only a single but significant change: they developed
:class:`~torch.nn.InstanceNorm2d` and used it as drop-in replacement for
:class:`~torch.nn.BatchNorm2d` layers. To account for this we provide ``instance_norm``
flag, which defaults to ``True``.

The original authors also use the same repository for both implementations and only
differentiate between them with the branches:

+------------------+---------------------+
| Paper            | Branch              |
+==================+=====================+
| :cite:`ULVL2016` | ``texture_nets_v1`` |
+------------------+---------------------+
| :cite:`UVL2017`  | ``master``          |
+------------------+---------------------+

.. _ulyanov_et_al_2016-impl_params:

Behavioral changes
------------------

.. seealso::
  :ref:`Paper implementations <impl_params>`

The following parts are affected:

  - :func:`~pystiche_papers.ulyanov_et_al_2016.content_transform`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.GramLoss`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.ConvBlock`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.level`,
  - :func:`~pystiche_papers.ulyanov_et_al_2016.Transformer`.


Hyper parameters
----------------

.. seealso::
  :ref:`Paper implementations <impl_params>`

Although there are four possible combinations for ``impl_params`` and ``instance_norm``
only three are listed below. Since the hyper-parameters in both papers are equal,
both combinations with ``ìmpl_params=False`` are equal and thus not reported.


:func:`~pystiche_papers.ulyanov_et_al_2016.content_loss`
````````````````````````````````````````````````````````

+------------------+---------------------+----------------------+-----------+
| Parameter        | ``impl_params`` / ``ìnstance_norm``                    |
+                  +---------------------+----------------------+-----------+
|                  | ``True`` / ``True`` | ``True`` / ``False`` | ``False`` |
+==================+=====================+======================+===========+
| ``layer``        | ``"relu4_2"``                                          |
+------------------+---------------------+----------------------+-----------+
| ``score_weight`` | ``1e0``             | ``6e-1``             | ``1e0``   |
+------------------+---------------------+----------------------+-----------+


:func:`~pystiche_papers.ulyanov_et_al_2016.style_loss`
``````````````````````````````````````````````````````

+-------------------+--------------------------------------------------+----------------------+--------------------------------------+
| Parameter         | ``impl_params``                                                                                                |
+                   +--------------------------------------------------+----------------------+--------------------------------------+
|                   | ``True`` / ``True``                              | ``True`` / ``False`` | ``False``                            |
+===================+==================================================+======================+======================================+
| ``layers``        | ``("relu1_1", "relu2_1", "relu3_1", "relu4_1")`` | ``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")`` |
+-------------------+--------------------------------------------------+----------------------+--------------------------------------+
| ``layer_weights`` | ``"sum"``                                                                                                      |
+-------------------+--------------------------------------------------+----------------------+--------------------------------------+
| ``score_weight``  | ``1e0``                                          | ``1e3``              | ``1e0``                              |
+-------------------+--------------------------------------------------+----------------------+--------------------------------------+


:func:`~pystiche_papers.ulyanov_et_al_2016.content_transform`
`````````````````````````````````````````````````````````````

+---------------+---------+
| Parameter     | Value   |
+===============+=========+
| ``edge_size`` | ``256`` |
+---------------+---------+

:func:`~pystiche_papers.ulyanov_et_al_2016.style_transform`
```````````````````````````````````````````````````````````

+------------------------+---------------------+----------------------+-----------+
| Parameter              | ``impl_params`` / ``ìnstance_norm``                    |
+                        +---------------------+----------------------+-----------+
|                        | ``True`` / ``True`` | ``True`` / ``False`` | ``False`` |
+========================+=====================+======================+===========+
| ``edge_size``          | ``256``                                                |
+------------------------+---------------------+----------------------+-----------+
| ``edge``               | ``"long"``                                             |
+------------------------+---------------------+----------------------+-----------+
| ``interpolation_mode`` | ``"bicubic"``       | ``"bilinear"``                   |
+------------------------+---------------------+----------------------+-----------+


:func:`~pystiche_papers.ulyanov_et_al_2016.batch_sampler`
`````````````````````````````````````````````````````````

+-----------------+---------------------+----------------------+-----------+
| Parameter       | ``impl_params`` / ``ìnstance_norm``                    |
+                 +---------------------+----------------------+-----------+
|                 | ``True`` / ``True`` | ``True`` / ``False`` | ``False`` |
+=================+=====================+======================+===========+
| ``num_batches`` | ``2_000``           | ``300``              | ``200``   |
+-----------------+---------------------+----------------------+-----------+
| ``batch_size``  | ``1``               | ``4``                | ``16``    |
+-----------------+---------------------+----------------------+-----------+

:func:`~pystiche_papers.ulyanov_et_al_2016.optimizer`
`````````````````````````````````````````````````````

+-----------+---------------------+----------------------+-----------+
| Parameter | ``impl_params`` / ``ìnstance_norm``                    |
+           +---------------------+----------------------+-----------+
|           | ``True`` / ``True`` | ``True`` / ``False`` | ``False`` |
+===========+=====================+======================+===========+
| ``lr``    | ``1e-3``            | ``1e0``              |           |
+-----------+---------------------+----------------------+-----------+


:func:`~pystiche_papers.ulyanov_et_al_2016.lr_scheduler`
````````````````````````````````````````````````````````

+--------------+---------------------+----------------------+-----------+
| Parameter    | ``impl_params`` / ``ìnstance_norm``                    |
+              +---------------------+----------------------+-----------+
|              | ``True`` / ``True`` | ``True`` / ``False`` | ``False`` |
+==============+=====================+======================+===========+
| ``lr_decay`` | ``0.8``                                    | ``0.7``   |
+--------------+---------------------+----------------------+-----------+
| ``delay``    | ``0``                                      | ``4``     |
+--------------+---------------------+----------------------+-----------+

Miscellaneous
`````````````

+----------------+---------------------+----------------------+-----------+
| Parameter      | ``impl_params`` / ``ìnstance_norm``                    |
+                +---------------------+----------------------+-----------+
|                | ``True`` / ``True`` | ``True`` / ``False`` | ``False`` |
+================+=====================+======================+===========+
| ``num_epochs`` | ``25``              | ``10``                           |
+----------------+---------------------+----------------------+-----------+


API
---

.. automodule:: pystiche_papers.ulyanov_et_al_2016

..
  _data.py
.. autofunction:: content_transform
.. autofunction:: style_transform
.. autofunction:: images
.. autofunction:: dataset
.. autofunction:: batch_sampler
.. autofunction:: image_loader

..
  _loss.py
.. autofunction:: content_loss
.. autoclass:: GramLoss
.. autofunction:: style_loss
.. autofunction:: perceptual_loss

..
  _modules.py
.. autofunction:: transformer

..
  _nst.py
.. autofunction:: training
.. autofunction:: stylization

..
  _utils.py

.. autofunction:: hyper_parameters
.. autofunction:: multi_layer_encoder
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: optimizer
.. autofunction:: lr_scheduler
