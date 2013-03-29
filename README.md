epac
====

Embarrassingly Parallel Array Computing. EPAC is a Machine Learning Workflow
builder.

Principles
----------

Combine Machine Learning tasks to build a workflow that may be executed in
sequential or in parallel:

- The composition of operations from the root to a leaf is a sequential pipeline.
- Internal nodes with several children (e.g.: folds of a cross-validation) lead
  to parallel execution of the children.

The execution is based on downstream/upstream data-flow paradigm:

- The top-down *downstream data-flow* is processed by Nodes from root to leaf nodes.
  Each node is identified by a unique *primary key*, it process the downstream
  data and pass it to its children. The downstream data-flow is a simple python
  dictionary. The final results are stored (*persistence*) by the leaves using a
  *intermediary key*.

- The bottom-up *upstream data-flow* start from results stored by leaves that 
  are locally reduced by nodes up to the tree's root. If no collision occurs
  in the upstream between intermediary keys results are simply moved up without
  interactions. If collision occurs, children results with similar intermediary key
  are aggregated (stacked) together. User may define a *reducer* that will be 
  applied to each (intermediary key, result) pair. Typically reducer will perform
  statistics (average over CV-folds, compute p-values over permutations) or will
  refit a model using the best arguments.


Application Programing interface
--------------------------------

- **Task**: a task is an object that implements 3 methods:
  - `fit(...)`
  - `predict(...)`
  - `score(...)`

- Node ::= Seq | ParMethods | ParGrid | ParCV | ParPerm
- Seq(Task, [Tasks]*)
- ParMethods(Task, [Tasks]*)
- ParGrid(Task, [Tasks]*)
- ParCV(Node|Task, n_folds, y, reducer)
- ParPerm(Node|Task, n_perms, y, permute, reducer)



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

