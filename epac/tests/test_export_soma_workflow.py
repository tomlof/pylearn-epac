#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on 2 May 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
@author: jinpeng.li@cea.fr

"""

import unittest
import os
import numpy as np
import shutil


def displayres(res_epac):

    for k1 in res_epac.keys():
        print ""+k1
        for k2 in res_epac[k1].keys():
            print "  "+k2
            print "    axis_name="+repr(res_epac[k1][k2].axis_name)
            print "    axis_values="+repr(res_epac[k1][k2].axis_values)
            for e3 in list(res_epac[k1][k2]):
                for e4 in list(e3):
                    print "      e4="+repr(e4)


def is_numeric_paranoid(obj):

    return isinstance(obj, (int, long, float, complex))


def isequal(array1, array2):

    if(isinstance(array1, dict)):
        for key in array1.keys():
            if not isequal(array1[key], array2[key]):
                return False

        return True

    array1 = np.asarray(list(array1))
    array2 = np.asarray(list(array2))

    for index in xrange(len(array1.flat)):

        if (
            (type(array1.flat[index]) is np.ndarray) or
            (type(array1.flat[index]) is list)
        ):
            return isequal(array1.flat[index], array2.flat[index])
        else:
            if (is_numeric_paranoid(array1.flat[index])):
                if (np.absolute(array1.flat[index] -
                   array2.flat[index]) > 0.00001):
                    return False
            else:
                if array1.flat[index] != array2.flat[index]:
                    return False
    return True


class EpacWorkflowTest(unittest.TestCase):

    def setUp(self):

    ########################################################################
    ## Input paths
        '''
        +my_epac_working_directory
          -epac_datasets.npz
          +storekeys
            +...
          -epac_workflow_example
        '''

        ## Setup a working directory (my_working_directory)
        self.my_working_directory = "/tmp/my_epac_working_directory"

        ## key_file and datasets_file should be ***RELATIVE*** path
        ## It is mandatory for mapping path in soma-workflow
        ## since my_working_directory will be changed on the cluster
        self.datasets_file = "./epac_datasets.npz"
        self.key_file = "./storekeys"
        self.soma_workflow_file = "./epac_workflow_example"

        #######################################################################
        ## Clean and change the working directory
        # so that we can use relative path in the directory
        # my_working_directory

        if os.path.isdir(self.my_working_directory):
            shutil.rmtree(self.my_working_directory)

        os.mkdir(self.my_working_directory)
        os.chdir(self.my_working_directory)

        ######################################################################
        ## DATASET
        from sklearn import datasets
        from sklearn.svm import SVC
        from sklearn.feature_selection import SelectKBest

        self.X, self.y = datasets.make_classification(
            n_samples=10, n_features=50,
            n_informative=5)

        np.savez(self.datasets_file, X=self.X, y=self.y)

        #######################################################################
        ## EPAC WORKFLOW
        # -------------------------------------
        #             ParPerm                Perm (Splitter)
        #         /     |       \
        #        0      1       2            Samples (Slicer)
        #        |
        #       ParCV                        CV (Splitter)
        #  /       |       \
        # 0        1       2                 Folds (Slicer)
        # |        |       |
        # Seq     Seq     Seq                Sequence
        # |
        # 2                                  SelectKBest (Estimator)
        # |
        # ParGrid
        # |                     \
        # SVM(linear,C=1)   SVM(linear,C=10)  Classifiers (Estimator)

        from epac import ParPerm, ParCV, Seq, ParGrid

        self.wf = None

        pipeline = Seq(SelectKBest(k=2),
                       ParGrid(*[SVC(kernel="linear", C=C) for C in [1, 10]]))

        self.wf = ParPerm(ParCV(pipeline, n_folds=3),
                          n_perms=10, permute="y", y=self.y)

        self.wf.save(store=self.key_file)

    def tearDown(self):
        pass

    def test_soma_workflow_cluster(self):

        from soma.workflow.client import Helper
        from epac.exports import export2somaworkflow

        (wf_id, controller) = export2somaworkflow(
            in_datasets_file=self.datasets_file,
            in_working_directory=self.my_working_directory,
            out_soma_workflow_file=self.soma_workflow_file,
            in_tree_root=self.wf,
            in_is_sumbit=True,
            in_resource_id="ed203246@gabriel",
            in_login="ed203246",
            in_pw=""
        )

        ## wait the workflow to finish
        Helper.wait_workflow(wf_id, controller)
        ## transfer the output files from the workflow
        Helper.transfer_output_files(wf_id, controller)
        controller.delete_workflow(wf_id)

        self.start2cmp()

    def test_soma_workflow(self):

        from soma.workflow.client import Helper
        from epac.exports import export2somaworkflow

        (wf_id, controller) = export2somaworkflow(
            in_datasets_file=self.datasets_file,
            in_working_directory=self.my_working_directory,
            out_soma_workflow_file=self.soma_workflow_file,
            in_tree_root=self.wf,
            in_is_sumbit=True,
            in_resource_id="",
            in_login="",
            in_pw=""
        )

        ## wait the workflow to finish
        Helper.wait_workflow(wf_id, controller)
        ## transfer the output files from the workflow
        Helper.transfer_output_files(wf_id, controller)
        controller.delete_workflow(wf_id)

        self.start2cmp()

    def test_soma_workflow_nodes(self):

        from soma.workflow.client import Helper
        from epac.exports import export2somaworkflow

        nodes = self.wf.get_node(regexp="*/ParPerm/*")

        (wf_id, controller) = export2somaworkflow(
            in_datasets_file=self.datasets_file,
            in_working_directory=self.my_working_directory,
            out_soma_workflow_file=self.soma_workflow_file,
            in_nodes=nodes,
            in_is_sumbit=True,
            in_resource_id="",
            in_login="",
            in_pw=""
        )

        ## wait the workflow to finish
        Helper.wait_workflow(wf_id, controller)
        ## transfer the output files from the workflow
        Helper.transfer_output_files(wf_id, controller)
        controller.delete_workflow(wf_id)

        self.start2cmp()

    def start2cmp(self):

        from epac.workflow.base import conf
        from epac import WF

        os.chdir(self.my_working_directory)

        ##### wf_key depends on your output in your map process
        wf_key = (
            conf.KEY_PROT_FS +
            conf.KEY_PROT_PATH_SEP +
            self.key_file +
            os.path.sep+os.walk(self.key_file).next()[1][0]
        )

        swf_wf = WF.load(wf_key)  # Results from soma-workflow
        res_swf = swf_wf.reduce()  # Reduce process

        ###################################################################
        ## Run without soma-workflow
        self.wf.fit_predict(X=self.X, y=self.y)
        res_epac = self.wf.reduce()

        self.compare_res(res_epac, res_swf)

    def compare_res(self, R1, R2):

        # displayres(R1)
        # displayres(R2)

        comp = dict()
        for key in R1.keys():

            r1 = R1[key]
            r2 = R2[key]

            comp[key] = True

#        for k in set(r1.keys()).intersection(set(r2.keys())):
#            comp[k]=True
#            if not isequal(r1[k],r2[k]):
#               comp[k]=False

            for k in r1.keys():
                comp[k] = True
                if not isequal(r1[k], r2[k]):
                    comp[k] = False

        for key in comp.keys():
            self.assertTrue(comp[key])

# return comp
# for key in comp:
#    for subkey in comp[key]:
#        self.assertTrue(comp[key][subkey],
#        u'Diff for key: "%s" and attribute: "%s"' % (key, subkey))

if __name__ == '__main__':
    unittest.main()
