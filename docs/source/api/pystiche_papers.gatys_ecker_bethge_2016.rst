``pystiche_papers.gatys_ecker_bethge_2016``
===========================================

+--------------------------+-----------------------------------------------------------+
| Title                    | Image Style Transfer Using Convolutional Neural Networks  |
+--------------------------+-----------------------------------------------------------+
| Authors                  | Leon A. Gatys, Alexander. S. Ecker, and Matthias Bethge   |
+--------------------------+-----------------------------------------------------------+
| Citation                 | :cite:`GEB2016`                                           |
+--------------------------+-----------------------------------------------------------+
| Reference implementation | |repo|_ / |archive|_                                      |
+--------------------------+-----------------------------------------------------------+
| Variant                  | Image optimization                                        |
+--------------------------+-----------------------------------------------------------+
| Content loss             | :class:`~pystiche.ops.FeatureReconstructionLoss`          |
+--------------------------+-----------------------------------------------------------+
| Style loss               | :class:`~pystiche.ops.GramLoss`                           |
+--------------------------+-----------------------------------------------------------+

.. |repo| replace:: Repository
.. _repo: https://github.com/leongatys/PytorchNeuralStyleTransfer

.. |archive| replace:: Archive
.. _archive: https://github.com/pmeier/PytorchNeuralStyleTransfer/tree/c673ff2dad4cebaf753aa94bf1658292d967058a

.. _gatys_ecker_bethge_2016-impl_params:

Behavioral changes
------------------

.. seealso::
  :ref:`Paper implementations <impl_params>`

The following parts are affected:

- :class:`~pystiche_papers.gatys_ecker_bethge_2016.FeatureReconstructionLoss`
- :class:`~pystiche_papers.gatys_ecker_bethge_2016.MultiLayerEncodingLoss`
- :class:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder`

Hyper parameters
----------------

.. seealso::
  :ref:`Paper implementations <impl_params>`

Empty cells mean, that the parameter is not defined in the paper or no default is set
in the reference implementation of the original authors. In both cases the available
value is used as default.


:func:`~pystiche_papers.gatys_ecker_bethge_2016.content_loss`
`````````````````````````````````````````````````````````````

+------------------+---------------+---------------+
| Parameter        | ``impl_params``               |
+                  +---------------+---------------+
|                  | ``True``      | ``False``     |
+==================+===============+===============+
| ``layer``        | ``"relu4_2"`` | ``"conv4_2"`` |
+------------------+---------------+---------------+
| ``score_weight`` | ``1e0``                       |
+------------------+-------------------------------+


:func:`~pystiche_papers.gatys_ecker_bethge_2016.style_loss`
```````````````````````````````````````````````````````````

+------------------+--------------------------------------------------------------+-------------------------------------------------------------+
| Parameter        | ``impl_params``                                                                                                            |
+                  +--------------------------------------------------------------+-------------------------------------------------------------+
|                  | ``True``                                                     | ``False``                                                   |
+==================+==============================================================+=============================================================+
| ``layers``        | ``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")`` | ``("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")`` |
+-------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| ``layer_weights`` | ``(2.4e-04, 6.1e-05, 1.5e-05, 3.8e-06, 3.8e-06)`` [#f1]_    | ``"mean"``                                                  |
+-------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| ``score_weight``  | ``1e3`` [#f2]_                                                                                                            |
+-------------------+-------------------------------------------------------------+-------------------------------------------------------------+


:func:`~pystiche_papers.gatys_ecker_bethge_2016.nst`
````````````````````````````````````````````````````

+--------------------+---------------+--------------+
| Parameter          | ``impl_params``              |
+                    +---------------+--------------+
|                    | ``True``      | ``False``    |
+====================+===============+==============+
| ``image_size``     | ``512``                      |
+--------------------+---------------+--------------+
| ``starting_point`` | ``"content"`` | ``"random"`` |
+--------------------+---------------+--------------+
| ``num_steps``      | ``500``       |              |
+--------------------+---------------+--------------+


.. [#f1]
  The ``layer_weights`` are computed by :math:`1 / n^2` where :math:`n` denotes the
  number of channels of a feature map from the corresponding layer in the
  :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder`.
.. [#f2]
  The paper also reports ``score_weight=1e-4`` for some images.

API
---

.. automodule:: pystiche_papers.gatys_ecker_bethge_2016

..
  _data.py
.. autofunction:: images

..
  _loss.py
.. autoclass:: FeatureReconstructionLoss
.. autofunction:: content_loss
.. autoclass:: MultiLayerEncodingLoss
.. autofunction:: style_loss
.. autofunction:: perceptual_loss

..
  _nst.py
.. autofunction:: nst

..
  _utils.py
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: optimizer
.. autofunction:: multi_layer_encoder
.. autofunction:: compute_layer_weights
.. autofunction:: hyper_parameters
