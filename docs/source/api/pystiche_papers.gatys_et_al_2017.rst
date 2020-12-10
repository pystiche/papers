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
| Content loss             | :class:`~pystiche.ops.FeatureReconstructionOperator`      |
+--------------------------+-----------------------------------------------------------+
| Style loss               | :class:`~pystiche.ops.GramOperator`                       |
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

- :class:`~pystiche_papers.gatys_et_al_2017.MultiLayerEncodingOperator`


Hyper parameters
----------------

:func:`~pystiche_papers.gatys_et_al_2017.content_loss`
``````````````````````````````````````````````````````

+------------------+---------------+
| Parameter        | Value         |
+==================+===============+
| ``layer``        | ``"relu4_2"`` |
+------------------+---------------+
| ``score_weight`` | ``1e0``       |
+------------------+---------------+


:func:`~pystiche_papers.gatys_et_al_2017.style_loss`
````````````````````````````````````````````````````

+-------------------+---------------------------------------------------------------+
| Parameter         | Value                                                         |
+===================+===============================================================+
| ``layers``        | ``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")``   |
+-------------------+---------------------------------------------------------------+
| ``layer_weights`` | ``(2.44e-04, 6.10e-05, 1.53e-05, 3.81e-06, 3.81e-06)`` [#f1]_ |
+-------------------+---------------------------------------------------------------+
| ``score_weight``  | ``1e3``                                                       |
+-------------------+---------------------------------------------------------------+


:func:`~pystiche_papers.gatys_et_al_2017.guided_style_loss`
```````````````````````````````````````````````````````````

+--------------------+---------------------------------------------------------------+
| Parameter          | Value                                                         |
+====================+===============================================================+
| ``layers``         | ``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")``   |
+--------------------+---------------------------------------------------------------+
| ``layer_weights``  | ``(2.44e-04, 6.10e-05, 1.53e-05, 3.81e-06, 3.81e-06)`` [#f1]_ |
+--------------------+---------------------------------------------------------------+
| ``region_weights`` | ``"sum"``                                                     |
+--------------------+---------------------------------------------------------------+
| ``score_weight``   | ``1e3``                                                       |
+--------------------+---------------------------------------------------------------+


.. [#f1]
  The ``layer_weights`` are not fixed, but are computed with
  :func:`~pystiche_papers.gatys_et_al_2017.compute_layer_weights`. The values here are
  the approximate result for the default ``layers`` and
  :func:`~pystiche_papers.gatys_et_al_2017.multi_layer_encoder`.

API
---

.. automodule:: pystiche_papers.gatys_et_al_2017

..
  _data.py
.. autofunction:: images

..
  _loss.py
.. autofunction:: content_loss
.. autoclass:: MultiLayerEncodingOperator
.. autofunction:: style_loss
.. autofunction:: guided_style_loss
.. autofunction:: perceptual_loss
.. autofunction:: guided_perceptual_loss

..
  _nst.py
.. autofunction:: nst
.. autofunction:: guided_nst

..
  _utils.py
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: multi_layer_encoder
.. autofunction:: optimizer
.. autofunction:: compute_layer_weights
.. autofunction:: hyper_parameters
