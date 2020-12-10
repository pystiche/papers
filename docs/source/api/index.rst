Package reference
=================

.. toctree::
  :maxdepth: 1

  pystiche_papers.data
  pystiche_papers.utils

.. _impl_params:

Paper implementations
---------------------

Unfortunately, the reference implementation of the original authors often deviates from
what is described in the paper. To account for this, the ``impl_params`` flag is used.
It defaults to ``True`` since the parameters in the implementation were probably used
to generate the results in the paper.

In general, the deviations can be separated into two classes:

1. **Behavioral changes**: These changes often result by misconceptions of the author
   for how their used framework or library works internally. An example for this is the
   usage of the mean squared error (MSE) in the reference implementation whereas the
   squared error (SE) is reported in the paper. In some cases these changes also
   account for completely undocumented behavior.

   These changes have hard-coded behavior and cannot be adapted freely, but rather only
   be switched between both sets.

2. **Hyper-parameter changes**: In contrast to 1., changes of the hyper-parameters are
   not hard-coded and can be freely adapted. Both sets of parameters can be accessed
   with the respective ``hyper_parameters`` functions.

You can find information on both types of changes for each paper implementation in the
respective "Behavioral changes" and "Hyper parameters" sections.

.. toctree::
  :maxdepth: 1

  pystiche_papers.gatys_ecker_bethge_2016
  pystiche_papers.gatys_et_al_2017
  pystiche_papers.johnson_alahi_li_2016
  pystiche_papers.li_wand_2016
  pystiche_papers.ulyanov_et_al_2016
