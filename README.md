epac
====

Embarrassingly Parallel Array Computing

Combine Machine Learning operation to build an execution tree of tasks that may
be executed in sequential or in parallel:
- The composition of operations from the root to a leaf is a sequential pipeline.
- Internal nodes with several children (e.g.: folds of a cross-validation) lead
  to parallel execution of the children.

The execution is based on a dataflow paradigm:
- the top-down (down)stream data-flow 
- bootum-up (up)stream : 


Persistence of map outputs

- Split = Resampling
- Tasks = Mapper = etimators

Tree = Pipelines


Data-flow
- top-down (down)stream split based on data-resampling
- bootum-up (up)stream : local reduce to combine local results

Splitters: process downstream data-flow.
They are non leaf node  (degree >= 1) with children.
They split the downstream data-flow to their children.
They reduce the upstream data-flow from their children, thus they are
also Reducer

Slicers: process downstream data-flow.
They are Splitters children.
They reslice the downstream data-flow.
They do nothing on the upstream data-flow.

Reducers: process upstream data-flow.

