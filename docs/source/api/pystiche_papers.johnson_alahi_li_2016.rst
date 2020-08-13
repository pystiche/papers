``pystiche_papers.johnson_alahi_li_2016``
=========================================

Unfortunately, the parameters in the reference implementation differ from the parameters
described in the paper. If ``impl_params is True``, the parameters from the reference
implementation are used instead of the parameters from the paper. The following
functions are affected by the parameter ``impl_params``:

  - the :func:`~pystiche_papers.johnson_alahi_li_2016.training`
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.stylization`
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.multi_layer_encoder`
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.content_transform`
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.style_transform`
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.batch_sampler`
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.transformer`
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.content_loss`
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.style_loss`
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.regularization`

.. _table-hyperparameters-johnson_alahi_li_2016:

The original authors provided models together with the used hyperparameters.
Unfortunately, the used hyperparameters deviate from the defaults given in the
reference implementation. If you specify a ``style`` and ``impl_params is True`` the
table below is used to determine

- the ``size`` the ``style_image`` is resized to with the
  :func:`~pystiche_papers.johnson_alahi_li_2016.style_transform`,

- the ``score_weight`` for

  - the :func:`~pystiche_papers.johnson_alahi_li_2016.content_loss`,
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.style_loss`,
  - the :func:`~pystiche_papers.johnson_alahi_li_2016.regularization`, as well as

- the ``num_batches`` of the
  :func:`~pystiche_papers.johnson_alahi_li_2016.image_loader`.


+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``style``             | ``instance_norm`` | ``size`` | ``score_weight``                                       | ``num_batches`` |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
|                       |                   |          | ``content_loss`` | ``style_loss`` | ``regularization`` |                 |
+=======================+===================+==========+==================+================+====================+=================+
| ``"candy"``           | ``True``          | ``384``  | ``1.0``          | ``10.0``       | ``1e-4``           | ``40000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``"composition_vii"`` | ``False``         | ``512``  | ``1.0``          | ``5.0``        | ``1e-6``           | ``60000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``"feathers"``        | ``True``          | ``180``  | ``1.0``          | ``10.0``       | ``1e-5``           | ``60000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``"la_muse"``         | ``False``         | ``512``  | ``1.0``          | ``5.0``        | ``1e-5``           | ``40000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``"la_muse"``         | ``True``          | ``512``  | ``0.5``          | ``10.0``       | ``1e-4``           | ``40000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``"mosaic"``          | ``True``          | ``512``  | ``1.0``          | ``10.0``       | ``1e-5``           | ``60000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``"starry_night"``    | ``False``         | ``512``  | ``1.0``          | ``3.0``        | ``1e-5``           | ``40000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``"the_scream"``      | ``True``          | ``384``  | ``1.0``          | ``20.0``       | ``1e-5``           | ``60000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``"the_wave"``        | ``False``         | ``512``  | ``1.0``          | ``5.0``        | ``1e-4``           | ``40000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+
| ``"udnie"``           | ``True``          | ``256``  | ``0.5``          | ``10.0``       | ``1e-6``           | ``40000``       |
+-----------------------+-------------------+----------+------------------+----------------+--------------------+-----------------+


.. automodule:: pystiche_papers.johnson_alahi_li_2016

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
.. autofunction:: regularization
.. autofunction:: style_loss
.. autofunction:: style_transform
.. autofunction:: stylization
.. autofunction:: training
.. autofunction:: transformer
