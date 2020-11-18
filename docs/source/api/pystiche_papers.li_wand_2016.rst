``pystiche_papers.li_wand_2016``
================================

+--------------------------+-----------------------------------------------------------+
| Title                    | Combining Markov Random Fields and Convolutional Neural   |
|                          |                                                           |
|                          | Networks for Image Synthesis                              |
+--------------------------+-----------------------------------------------------------+
| Authors                  | Chuan Li and Michael Wand                                 |
+--------------------------+-----------------------------------------------------------+
| Citation                 | :cite:`LW2016`                                            |
+--------------------------+-----------------------------------------------------------+
| Reference implementation | |repo|_ / |archive|_                                      |
+--------------------------+-----------------------------------------------------------+
| Variant                  | Image optimization                                        |
+--------------------------+-----------------------------------------------------------+
| Content loss             | :class:`~pystiche.ops.FeatureReconstructionOperator`      |
+--------------------------+-----------------------------------------------------------+
| Style loss               | :class:`~pystiche.ops.MRFOperator`                        |
+--------------------------+-----------------------------------------------------------+
| Regularization           | :class:`~pystiche.ops.TotalVariationOperator`             |
+--------------------------+-----------------------------------------------------------+

.. |repo| replace:: Repository
.. _repo: https://github.com/chuanli11/CNNMRF/

.. |archive| replace:: Archive
.. _archive: https://github.com/pmeier/CNNMRF/tree/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f

.. _table-li_wand_2016-impl_params:

Implementation parameters
-------------------------

Unfortunately, the the behavior in the reference implementation differ from what is
described in the paper. The following parts are affected:

- :class:`~pystiche_papers.li_wand_2016.FeatureReconstructionOperator`
- :class:`~pystiche_papers.li_wand_2016.MRFOperator`
- :class:`~pystiche_papers.li_wand_2016.TotalVariationOperator`
- :class:`~pystiche_papers.li_wand_2016.target_transforms`

The behavior can be switched with the ``impl_params`` flag, which defaults to ``True``.

Hyper parameters
----------------

Furthermore, the :func:`~pystiche_papers.li_wand_2016.hyper_parameters` also mismatch.
The ``impl_params`` flag is used to switch between between these two sets.


:func:`~pystiche_papers.li_wand_2016.content_loss`
``````````````````````````````````````````````````

+------------------+-----------------+-----------+
| Parameter        | ``impl_params``             |
+                  +-----------------+-----------+
|                  | ``True``        | ``False`` |
+==================+=================+===========+
| ``layer``        | ``"relu4_2"``               |
+------------------+-----------------+-----------+
| ``score_weight`` | ``2e1``         | ``1e0``   |
+------------------+-----------------+-----------+


:func:`~pystiche_papers.li_wand_2016.target_transforms`
```````````````````````````````````````````````````````

+-----------------------+-----------------+-----------+
| Parameter             | ``impl_params``             |
+                       +-----------------+-----------+
|                       | ``True``        | ``False`` |
+=======================+=================+===========+
| ``num_scale_steps``   | ``0``           | ``3``     |
+-----------------------+-----------------+-----------+
| ``scale_step_width``  | ``5e-2``                    |
+-----------------------+-----------------+-----------+
| ``num_rotate_steps``  | ``0``           | ``2``     |
+-----------------------+-----------------+-----------+
| ``rotate_step_width`` | ``7.5``                     |
+-----------------------+-----------------+-----------+

:func:`~pystiche_papers.li_wand_2016.style_loss`
````````````````````````````````````````````````

+-------------------+-----------------+-----------+
| Parameter         | ``impl_params``             |
+                   +-----------------+-----------+
|                   | ``True``        | ``False`` |
+===================+=================+===========+
| ``layers``        | ``("relu3_1", "relu4_1")``  |
+-------------------+-----------------+-----------+
| ``layer_weights`` | ``"sum"``                   |
+-------------------+-----------------+-----------+
| ``patch_size``    | ``3``                       |
+-------------------+-----------------+-----------+
| ``stride``        | ``2``           | ``1``     |
+-------------------+-----------------+-----------+
| ``score_weight``  | ``1e-4``        | ``1e0``   |
+-------------------+-----------------+-----------+


:func:`~pystiche_papers.li_wand_2016.regularization`
````````````````````````````````````````````````````

+------------------+-----------------+-----------+
| Parameter        | ``impl_params``             |
+                  +-----------------+-----------+
|                  | ``True``        | ``False`` |
+==================+=================+===========+
| ``score_weight`` | ``1e-3``                    |
+------------------+-----------------+-----------+


:func:`~pystiche_papers.li_wand_2016.image_pyramid`
````````````````````````````````````````````````````

+-------------------+-----------------+-----------+
| Parameter         | ``impl_params``             |
+                   +-----------------+-----------+
|                   | ``True``        | ``False`` |
+===================+=================+===========+
| ``max_edge_size`` | ``384``                     |
+-------------------+-----------------+-----------+
| ``num_steps``     | ``100``         | ``200``   |
+-------------------+-----------------+-----------+
| ``num_levels``    | ``3``           | ``None``  |
+-------------------+-----------------+-----------+
| ``min_edge_size`` | ``64``                      |
+-------------------+-----------------+-----------+
| ``edge``          | ``"long"``                  |
+-------------------+-----------------+-----------+

``num_levels=None`` implies that the number of levels is automatically calculated
depending on ``max_edge_size`` and ``min_edge_size``. See
:class:`pystiche.pyramid.OctaveImagePyramid` for details.


API
---


.. automodule:: pystiche_papers.li_wand_2016

..
  _data.py
.. autofunction:: images

..
  _loss.py
.. autoclass:: FeatureReconstructionOperator
.. autofunction:: content_loss
.. autoclass:: MRFOperator
.. autofunction:: style_loss
.. autoclass:: TotalVariationOperator
.. autofunction:: regularization
.. autofunction:: perceptual_loss

..
  _nst.py
.. autofunction:: nst

..
  _pyramid.py
.. autofunction:: image_pyramid

..
  _utils.py
.. autofunction:: hyper_parameters
.. autofunction:: extract_normalized_patches2d
.. autofunction:: target_transforms
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: multi_layer_encoder
.. autofunction:: optimizer
