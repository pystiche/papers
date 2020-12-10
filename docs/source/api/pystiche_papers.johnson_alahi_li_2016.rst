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

.. _johnson_alahi_li_2016-impl_params:

Behavioral changes
------------------

.. seealso::
  :ref:`Paper implementations <impl_params>`

The following parts are affected:

- :func:`~pystiche_papers.johnson_alahi_li_2016.content_transform`
- :class:`~pystiche_papers.johnson_alahi_li_2016.GramOperator`
- :class:`~pystiche_papers.johnson_alahi_li_2016.TotalVariationOperator`
- :func:`~pystiche_papers.johnson_alahi_li_2016.decoder`
- :func:`~pystiche_papers.johnson_alahi_li_2016.preprocessor`
- :func:`~pystiche_papers.johnson_alahi_li_2016.postprocessor`
- :func:`~pystiche_papers.johnson_alahi_li_2016.multi_layer_encoder`


Hyper parameters
----------------

.. seealso::
  :ref:`Paper implementations <impl_params>`


:func:`~pystiche_papers.johnson_alahi_li_2016.content_loss`
```````````````````````````````````````````````````````````

+------------------+-----------------+-----------+
| Parameter        | ``impl_params``             |
+                  +-----------------+-----------+
|                  | ``True``        | ``False`` |
+==================+=================+===========+
| ``layer``        | ``"relu2_2"``               |
+------------------+-----------------+-----------+
| ``score_weight`` |                 | ``1e0``   |
+------------------+-----------------+-----------+


:func:`~pystiche_papers.johnson_alahi_li_2016.style_loss`
`````````````````````````````````````````````````````````

+-------------------+-----------------+---------------------------------+
| Parameter         | ``impl_params``                                   |
+                   +-----------------+---------------------------------+
|                   | ``True``        | ``False``                       |
+===================+=================+=================================+
| ``layers``        | ``("relu1_2", "relu2_2", "relu3_3", "relu4_3")``  |
+-------------------+-----------------+---------------------------------+
| ``layer_weights`` | ``"sum"``                                         |
+-------------------+-----------------+---------------------------------+
| ``score_weight``  |                 | ``5e0``                         |
+-------------------+-----------------+---------------------------------+


:func:`~pystiche_papers.johnson_alahi_li_2016.regularization`
`````````````````````````````````````````````````````````````

+------------------+-----------------+-----------+
| Parameter        | ``impl_params``             |
+                  +-----------------+-----------+
|                  | ``True``        | ``False`` |
+==================+=================+===========+
| ``score_weight`` |                 |``1e-6``   |
+------------------+-----------------+-----------+


:func:`~pystiche_papers.johnson_alahi_li_2016.content_transform`
````````````````````````````````````````````````````````````````

+----------------+-----------------+-----------+
| Parameter      | ``impl_params``             |
+                +-----------------+-----------+
|                | ``True``        | ``False`` |
+================+=================+===========+
| ``image_size`` | ``(256, 256)``              |
+----------------+-----------------+-----------+


:func:`~pystiche_papers.johnson_alahi_li_2016.style_transform`
``````````````````````````````````````````````````````````````

+---------------+-----------------+-----------+
| Parameter     | ``impl_params``             |
+               +-----------------+-----------+
|               | ``True``        | ``False`` |
+===============+=================+===========+
| ``edge_size`` | ``256``                     |
+---------------+-----------------+-----------+
| ``edge``      | ``"long"``                  |
+---------------+-----------------+-----------+


:func:`~pystiche_papers.johnson_alahi_li_2016.batch_sampler`
````````````````````````````````````````````````````````````

+-----------------+-----------------+-----------+
| Parameter       | ``impl_params``             |
+                 +-----------------+-----------+
|                 | ``True``        | ``False`` |
+=================+=================+===========+
| ``num_batches`` | ``40000``                   |
+-----------------+-----------------+-----------+
| ``batch_size``  | ``4``                       |
+-----------------+-----------------+-----------+


API
---

.. automodule:: pystiche_papers.johnson_alahi_li_2016

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
.. autoclass:: GramOperator
.. autofunction:: style_loss
.. autoclass:: TotalVariationOperator
.. autofunction:: regularization
.. autofunction:: perceptual_loss

..
  _modules.py
.. autoclass:: Transformer
.. autofunction:: transformer

..
  _nst.py
.. autofunction:: training
.. autofunction:: stylization

..
  _utils.py
.. autofunction:: hyper_parameters
.. autofunction:: preprocessor
.. autofunction:: postprocessor
.. autofunction:: multi_layer_encoder
.. autofunction:: optimizer
