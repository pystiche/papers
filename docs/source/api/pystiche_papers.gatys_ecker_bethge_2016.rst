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
| Content loss             | :class:`~pystiche.ops.FeatureReconstructionOperator`      |
+--------------------------+-----------------------------------------------------------+
| Style loss               | :class:`~pystiche.ops.GramOperator`                       |
+--------------------------+-----------------------------------------------------------+

.. |repo| replace:: Repository
.. _repo: https://github.com/leongatys/PytorchNeuralStyleTransfer

.. |archive| replace:: Archive
.. _archive: https://github.com/pmeier/PytorchNeuralStyleTransfer/tree/c673ff2dad4cebaf753aa94bf1658292d967058a

.. _table-gatys_ecker_bethge_2016-impl_params:

Behavioral changes
------------------

.. seealso::
  :ref:`Paper implementations <impl_params>`

The following parts are affected:

- :class:`~pystiche_papers.li_wand_2016.FeatureReconstructionOperator`
- :class:`~pystiche_papers.li_wand_2016.MultiLayerEncodingOperator`
- :class:`~pystiche_papers.li_wand_2016.multi_layer_encoder`

Hyper parameters
----------------

Empty cells mean, that the parameter is not defined in the paper or no default is set
in the reference implementation of the original authors. In both cases the available
value is used as default.


:func:`~pystiche_papers.gatys_ecker_bethge_2016.content_loss`
`````````````````````````````````````````````````````````````

+------------------+---------------+
| Parameter        | Value         |
+==================+===============+
| ``layer``        | ``"relu4_2"`` |
+------------------+---------------+
| ``score_weight`` | ``1e0``       |
+------------------+---------------+


:func:`~pystiche_papers.gatys_ecker_bethge_2016.style_loss`
```````````````````````````````````````````````````````````

+-------------------+-------------------------------------------------------------+
| Parameter         | Value                                                       |
+===================+=============================================================+
| ``layers``        | ``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")`` |
+-------------------+-------------------------------------------------------------+
| ``layer_weights`` | ``(2.44e-04, 6.10e-05, 1.53e-05, 3.81e-06, 3.81e-06)``      |
+-------------------+-------------------------------------------------------------+
| ``score_weight``  | ``1e3``                                                     |
+-------------------+-------------------------------------------------------------+

The ``layer_weights`` are not fixed, but are computed with
:func:`~pystiche_papers.gatys_ecker_bethge_2016.compute_layer_weights`. The values
above are the approximate result for the default ``layers`` and
:func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder`.


:func:`~pystiche_papers.gatys_ecker_bethge_2016.nst`
````````````````````````````````````````````````````

+--------------------+---------------+--------------+
| Parameter          | ``impl_params``              |
+                    +---------------+--------------+
|                    | ``True``      | ``False``    |
+====================+=================+============+
| ``starting_point`` | ``"content"`` | ``"random"`` |
+--------------------+---------------+--------------+
| ``num_steps``      | ``500``       |              |
+--------------------+---------------+--------------+


Miscellaneous
`````````````

+----------------+----------+-----------+
| Parameter      | ``impl_params``      |
+                +----------+-----------+
|                | ``True`` | ``False`` |
+================+==========+===========+
| ``image_size`` | ``500``  |           |
+----------------+----------+-----------+

API
---

.. automodule:: pystiche_papers.gatys_ecker_bethge_2016

..
  _data.py
.. autofunction:: images

..
  _loss.py
.. autoclass:: FeatureReconstructionOperator
.. autofunction:: content_loss
.. autoclass:: MultiLayerEncodingOperator
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
