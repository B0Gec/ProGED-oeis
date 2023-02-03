import os, sys
import numpy as np
import pickle
import ProGED as pg
from src.generate_data.systems_collection import strogatz, mysystems

struct_version = "s4"
path_main = "D:\\Experiments\\MLJ23\\"
path_base_out = f"{path_main}results\\proged\\sysident_num\\structures\\{struct_version}\\"
systems = {**strogatz, **mysystems}
all_grammars = [systems[i].grammar_type for i in list(systems.keys())]
unique_grammars = [list(x) for x in set(tuple(x) for x in all_grammars)]
grammars = [val for sublist in unique_grammars for val in sublist]

# ib, sys_name, igram, gram = 0, "myvdp", 0, grammars[0]
for sys_name in list(systems.keys()):
    for igram, gram in enumerate(grammars):

        if sys_name not in ['stlvdpBL']:
            print(f"skiping {sys_name} | {gram}: not intended to be done.")
            continue
        if not gram in systems[sys_name].grammar_type:
            print(f"System {sys_name} doesn't have the grammar {gram}. Skip.")
            continue


        print(f"Start: {sys_name} | {gram}")
        num_samples = systems[sys_name].num_samples
        num_batches = systems[sys_name].num_batches
        struct_filename = f"structs_{struct_version}_{gram}_nsamp{num_samples}_nbatch{num_batches}"
        batch_path = f"{path_base_out}{struct_filename}{os.sep}"

        # log the info
        old_stdout = sys.stdout
        log_file = open(f"{batch_path}{struct_filename}_{sys_name}_check_log.log", "w")
        sys.stdout = log_file

        for ib in range(num_batches):

            print("---------------------------------------------------------------------")
            print(f"grammar: {gram} | num_samples: {num_samples} | batch: {ib} | system: {sys_name} | V: {systems[sys_name].grammar_vars}")
            print("---------------------------------------------------------------------")

            # get structures
            batch_filename = f"{struct_filename}_b{ib}.pg"
            with open(os.path.join(batch_path, batch_filename), "rb") as file:
                structures = pickle.load(file)

            ## include the right system (to check if the search works)
            #structures.add_model("y", symbols={"x": ["x", "y", "u", "v"], "const": "C"})
            #structures.add_model("C*x + C*y*(1 - x**2)", symbols={"x": ["x", "y", "u", "v"], "const": "C"})

            expr_parts_yes = systems[sys_name].expr_parts_yes
            expr_parts_no = systems[sys_name].expr_parts_no

            # for each structure:
            # idx_structure, structure = 0, structures[0]
            # idim, idimopt = 0, 0
            for idx_structure, structure in enumerate(structures):
                for idim in range(len(expr_parts_yes)):
                    for idimopt in range(len(expr_parts_yes[idim])):
                        if all([expr_parts_yes[idim][idimopt][i] in str(structure.expr) for i in range(len(expr_parts_yes[idim][idimopt]))]) and \
                            not any([expr_parts_no[idim][j] in str(structure.expr) for j in range(len(expr_parts_no[idim]))]):
                            print(f"{sys_name} | dim: {idim+1} | idx: {idx_structure} | expr: {structure.expr}")


        sys.stdout = old_stdout
        log_file.close()