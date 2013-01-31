epac
====

Embarrassingly Parallel Array Computing

Principles
----------

Combine Machine Learning operations to build an execution tree of tasks that may
be executed in sequential or in parallel:

- The composition of operations from the root to a leaf is a sequential pipeline.

- Internal nodes with several children (e.g.: folds of a cross-validation) lead
  to parallel execution of the children.

The execution is based on dataflow paradigm:

- The top-down "downstream" data-flow is processed by "Mappers" from root to leaf node. 
  The final outputs are stored using a unique (primary) key. Those outputs are
  the input of the "upstream" flow. Typically, data blocs will be splitted into
  train and test sets, filtered and predictions will be produced by the leaves.

- The bootum-up "upstream" data-flow: reduces (locally) and combine results up
  to the tree's root. Typically, predictions will be concatenated or summarized
  by computing scores from leaves to root.

Key
---

Persistence
-----------

- Split = Resampling
- Tasks = Mapper = etimators

Tree = Pipelines


Data-flow
- top-down (down)stream split based on data-resampling
- bootum-up (up)stream : local reduce to combine local results

Details
-------

Nodes are of three types Mapper, Splitter or Slicer.

Splitters: process downstream data-flow.
They are non leaf node  (degree >= 1) with several children.
They split the downstream data-flow to their children.
They reduce the upstream data-flow from their children, thus they are
also Reducer

Slicers: process downstream data-flow.
They are Splitters children.
They reslice the downstream data-flow.
They do nothing on the upstream data-flow.

Reducers: process upstream data-flow.

