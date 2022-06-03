``pystiche_papers.gatys_et_al_2017``
====================================

+--------------------------+-----------------------------------------------------------+
| Title                    | Controlling Perceptual Factors in Neural Style Transfer   |
+--------------------------+-----------------------------------------------------------+
| Authors                  | Leon A. Gatys, Alexander. S. Ecker, Matthias Bethge,      |
|                          |                                                           |
|                          | Aaron Hertzmann, and Eli Shechtman                        |
+--------------------------+-----------------------------------------------------------+
| Citation                 | :cite:`GEB+2017`                                          |
+--------------------------+-----------------------------------------------------------+
| Reference implementation | |repo|_ / |archive|_                                      |
+--------------------------+-----------------------------------------------------------+
| Variant                  | Image optimization                                        |
+--------------------------+-----------------------------------------------------------+
| Content loss             | :class:`~pystiche.loss.FeatureReconstructionLoss`         |
+--------------------------+-----------------------------------------------------------+
| Style loss               | :class:`~pystiche.loss.GramLoss`                          |
+--------------------------+-----------------------------------------------------------+

.. |repo| replace:: Repository
.. _repo: https://github.com/leongatys/NeuralImageSynthesis/

.. |archive| replace:: Archive
.. _archive: https://github.com/pmeier/NeuralImageSynthesis/tree/cced0b978fe603569033b2c7f04460839e4d82c4

.. _gatys_et_al_2017-impl_params:

Behavioral changes
------------------

.. seealso::
  :ref:`Paper implementations <impl_params>`

The following parts are affected:

- :class:`~pystiche_papers.gatys_et_al_2017.MultiLayerEncodingLoss`


Hyper parameters
----------------

:func:`~pystiche_papers.gatys_et_al_2017.content_loss`
``````````````````````````````````````````````````````

+------------------+---------------+---------------+
| Parameter        | ``impl_params``               |
+                  +---------------+---------------+
|                  | ``True``      | ``False``     |
+==================+===============+===============+
| ``layer``        | ``"relu4_2"`` | ``"conv4_2"`` |
+------------------+---------------+---------------+
| ``score_weight`` | ``1e0``                       |
+------------------+-------------------------------+


:func:`~pystiche_papers.gatys_et_al_2017.style_loss`
````````````````````````````````````````````````````

+-------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| Parameter         | ``impl_params``                                                                                                           |
+                   +-------------------------------------------------------------+-------------------------------------------------------------+
|                   | ``True``                                                    | ``False``                                                   |
+===================+=============================================================+=============================================================+
| ``layers``        | ``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")`` | ``("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")`` |
+-------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| ``layer_weights`` | ``(2.4e-04, 6.1e-05, 1.5e-05, 3.8e-06, 3.8e-06)`` [#f1]_                                                                  |
+-------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| ``score_weight``  | ``1e3`` [#f1]_                                                                                                            |
+-------------------+-------------------------------------------------------------+-------------------------------------------------------------+


:func:`~pystiche_papers.gatys_et_al_2017.guided_style_loss`
```````````````````````````````````````````````````````````

+--------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| Parameter          | ``impl_params``                                                                                                           |
+                    +-------------------------------------------------------------+-------------------------------------------------------------+
|                    | ``True``                                                    | ``False``                                                   |
+====================+=============================================================+=============================================================+
| ``layers``         | ``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")`` | ``("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")`` |
+--------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| ``layer_weights``  | ``(2.4e-04, 6.1e-05, 1.5e-05, 3.8e-06, 3.8e-06)`` [#f1]_ [#f2]_                                                           |
+--------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| ``region_weights`` |                                                             | ``"sum"``                                                   |
+--------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| ``score_weight``   | ``1e3`` [#f1]_                                                                                                            |
+--------------------+-------------------------------------------------------------+-------------------------------------------------------------+

:func:`~pystiche_papers.gatys_et_al_2017.image_pyramid`
```````````````````````````````````````````````````````

+----------------+------------------------+-----------------+
| Parameter      | ``impl_params``                          |
+                +------------------------+-----------------+
|                | ``True``               | ``False``       |
+================+========================+=================+
| ``edge_sizes`` | ``(500, 1024)`` [#f3]_ | ``(512, 1024)`` |
+----------------+------------------------+-----------------+
| ``num_steps``  | [#f4]_                 | ``(500, 200)``  |
+----------------+------------------------+-----------------+


.. [#f1]
  The values are reported in the
  `supplementary material <http://bethgelab.org/media/uploads/stylecontrol/supplement/SupplementaryMaterial.pdf>`_.
.. [#f2]
  The ``layer_weights`` are computed by :math:`1 / n^2` where :math:`n` denotes the
  number of channels of a feature map from the corresponding layer in the
  :func:`~pystiche_papers.gatys_et_al_2017.multi_layer_encoder`.
.. [#f3]
  The paper only reports the ``edge_size`` for the low resolution.
.. [#f4]
  The paper only reports the ratio. i.e. :math:`500 / 200 = 2.5` of ``num_steps``.

API
---

.. automodule:: pystiche_papers.gatys_et_al_2017

..
  _data.py
.. autofunction:: images

..
  _loss.py
.. autofunction:: content_loss
.. autoclass:: MultiLayerEncodingLoss
.. autofunction:: style_loss
.. autofunction:: guided_style_loss
.. autofunction:: perceptual_loss
.. autofunction:: guided_perceptual_loss

..
  _nst.py
.. autofunction:: nst
.. autofunction:: guided_nst

..
  _pyramid.py
.. autofunction:: image_pyramid

..
  _utils.py
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: multi_layer_encoder
.. autofunction:: optimizer
.. autofunction:: compute_layer_weights
.. autofunction:: hyper_parameters
