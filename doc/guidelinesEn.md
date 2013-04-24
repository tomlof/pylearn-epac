Architecture
============

- Separate components (for example, soma-workflow, pyanatomist, vip etc.). Those can be achieved by adding interfaces for each component.
- It is better to build the interface according to the requirements of application.
- To help separate components, it is better to image that each component can have different implementations and each external dependencies can be replaced.


General guideline
==================

- Function is better to be shorter (there may be problems since 25 lines)
- Discriminant names are preferable for functions, classes, files or a set of files. Because it is easy to debug and rename.
- Limit your comments. It is better to make functions or code more clearly by using an explicate name rather than adding a comment.
- Some comments are very important, for example opition explanation.
- Use standard comments which are easy to search and which are adequate for : TODO, FIXME, XXX
- Boy scout rule: Leave the campground cleaner than you found it. 
- Write your method from top to down. For example: if a method A use a method B, write A and then B.
- Class order: top -> down, code client -> code interne (???), high level code -> low level code.
- Don't leave prints (proper stdout)


Python code format
=====================

- Pep 8 
- Recommandly donot go beyond 80 characters per line, strictly donot go beyond 120 characters.
- A module contains 400 lines in maximum
- All the members of an instance should be explictly declared in the function "__init__" or in the methods using the prefixs by "_init_" or called as constructor (don't add it just-in-time)
- No "import *"
- No relative imports
- Declare import standard python modules, and then add one empty line and the other externe imports (for example, aims, soma_workflow) and at the end import internal modules.
- Use a coherent name with the module for "import something as smthg"
- The private classes are renamed using the prefix double underscore (not yet applied). The protected method use the prefix by one underscore. ??? Pour l’accès par réference const (modification de l’objet autorisé mais pas de la référence), utiliser les properties.???
- The methods "friends" are should use the prefix with «_friend» in order to show which are protected but accessible for the architecture requirements of other classes, but not users.
- Data with list type is defined use a name with plural form.
- To distinguish the file name with path or not, relative or absolute, please use the belowing convention:
    * filename : the file name without the path (basename)
    * filepath : the file name with absolute path 
    * file_relative_path : the file name with a relative path


Commit Message
==============

Git format
-------------

- a synthetic line should start with the below tags
- a blank (empty) line will be ignored by git
- a comment can be described by serveral lines 

Tag list for the first line
---------------------------

- **ENH**: When adding or improving an existing or new class in term of capabilities,
- **COMP**: When fixing a compilation error or warning,
- **DOC**: When starting or improving the class documentation,
- **STYLE**: When enhancing the comments or coding style without any effect on the class behaviour,
- **REFAC**: When refactoring without adding new capabilities,
- **BUG**: When fixing a bug (if the bug is identified in the tracker, please add the reference),
- **INST**: When fixing issues related to installation,
- **PERF**: When improving performance,
- **TEST**: When adding or modifying a test,
- **WRG**: When correcting a warning.


