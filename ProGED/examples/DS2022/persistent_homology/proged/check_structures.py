import os, sys
import numpy as np
import pickle
import ProGED as pg
from proged.code.MLJ_helper_functions import get_expr_parts

exp_version = "v12"
path_main = "D:\\Experiments\\MLJ23\\proged\\numdiff"
path_base_out = f"{path_main}\\structures\\{exp_version}\\"

#grammars = ["polybrackets", "polystricter", "phaseosc"]
grammars = ["polynomial", "phaseosc"]
types = ["onestate", "lorenz", "twophase", "twostate1", "twostate2"]
types = ["twostate2"]
#grammar = sys.argv[1]
#itype = sys.argv[2]
num_samples = 10000
n_batches = 20

# grammar, itype = ["polynomial", "twostate1"]
for grammar in grammars:
    for itype in types:

        if (grammar == 'phaseosc') & (itype != 'twophase'):
            continue
        elif (itype == 'twophase') & (grammar != "phaseosc") & (grammar != "universal"):
            continue

        system_classification = {
            'onestate': ['vdp', 'stl', 'lotka'],
            'lorenz': ['lorenz'],
            'twophase': ['cphase'],
            'twostate1':['vdpvdpUS', 'stlvdpBL'],
            'twostate2': ['vdpvdpUS', 'stlvdpBL']
        }
        batch_path = f"{path_base_out}batchsize{n_batches}\\{itype}\\"

        # log the info
        old_stdout = sys.stdout
        log_file = open(f"{batch_path}structures_{exp_version}_{grammar}_{itype}_n{num_samples}_check_log.log","w")
        sys.stdout = log_file

        ## check if the structure is included for specific system
        # sys_name = "vdpvdpUS"
        for sys_name in system_classification[itype]:
            # ib = 0
            for ib in range(n_batches):

                print("---------------------------------------------------------------------")
                print(f"{grammar} | {itype} | {num_samples} | batch: {ib} | {sys_name}")
                print("---------------------------------------------------------------------")

                # get structures
                batch_filename = "structures_{}_{}_{}_n{}_batch{}.pg".format(exp_version,grammar,itype,num_samples, str(ib))
                with open(os.path.join(batch_path, batch_filename), "rb") as file:
                    structures = pickle.load(file)

                ## include the right system (to check if the search works)
                #structures.add_model("y", symbols={"x": ["x", "y", "u", "v"], "const": "C"})
                #structures.add_model("C*x + C*y*(1 - x**2)", symbols={"x": ["x", "y", "u", "v"], "const": "C"})

                sys_expr_parts, not_sys_expr_parts = get_expr_parts(sys_name)

                # for each structure:
                # idx_structure, structure = 0, structures[0]
                # idim, idimopt = 0, 0
                for idx_structure, structure in enumerate(structures):
                    for idim in range(len(sys_expr_parts)):
                        for idimopt in range(len(sys_expr_parts[idim])):
                            if all([sys_expr_parts[idim][idimopt][i] in str(structure.expr) for i in range(len(sys_expr_parts[idim][idimopt]))]) and \
                                not any([not_sys_expr_parts[idim][j] in str(structure.expr) for j in range(len(not_sys_expr_parts[idim]))]):
                                print(f"{sys_name} | dim: {idim+1} | idx: {idx_structure} | expr: {structure.expr}")


        sys.stdout = old_stdout
        log_file.close()