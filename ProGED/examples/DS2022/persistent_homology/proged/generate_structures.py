import os
import io
import pickle
import pandas as pd
import numpy as np
import ProGED as pg
from ProGED.generate import generate_models
from utils.generate_data_ODE_systems import generate_ODE_data
from ProGED.generators.grammar_construction import construct_production, grammar_from_template

def create_sh_file(**batch_settings):
    sys_name = batch_settings["system_type"]
    job_vers = batch_settings["exp_version"]
    nbatches = batch_settings["n_batches"]
    pyfile_name = "slurm_run_batch_{}_v{}.py".format(sys_name, job_vers)

    title = "slurm_run_batch_{}_v{}.sh".format(sys_name, job_vers)
    f = io.open(os.path.join(batch_settings["path_out"], title), "w", newline='\n')
    f.write("#!/bin/bash\n"
            "#SBATCH --job-name={}v{}\n".format(sys_name, job_vers))
    f.write("#SBATCH --time=2-00:00:00\n"
            "#SBATCH --mem-per-cpu=2GB\n")
    f.write("#SBATCH --array=0-{}\n".format(str(nbatches-1)))
    f.write("#SBATCH --cpus-per-task=1\n")
    f.write("#SBATCH --output=jobs/{}/v{}/nbatches{}/slurm_output_%A_%a.out\n".format(sys_name, job_vers, str(nbatches)))
    f.write("\nsingularity exec proged_container.sif python3.7 " + pyfile_name + " ${SLURM_ARRAY_TASK_ID}")
    f.close()


def create_batches(**batch_settings):

    np.random.seed(1)
    # get grammar
    if batch_settings["grammar"] == "universal":
        grammar = grammar_from_template("universal",
                                        {"variables": batch_settings["variables"],
                                         "p_vars": batch_settings["p_vars"]})

    elif batch_settings["grammar"] == "polynomial":
        grammarstr = construct_production(left="S", items=["S '+' T", "T"], probs=[0.75, 0.25])
        grammarstr += construct_production(left="T", items=["T '*' V", "'C'"], probs=[0.2, 0.8])
        grammarstr += construct_production(left="V", items=batch_settings["variables"], probs=batch_settings["p_vars"])
        grammar = pg.GeneratorGrammar(grammarstr)

    elif batch_settings["grammar"] == "polybrackets":
        grammarstr = construct_production("E", ["E '+' F", "E '-' F", "F"], [0.3, 0.3, 0.4])
        grammarstr += construct_production("F", ["F '*' T", "T"], [0.2, 0.8])
        grammarstr += construct_production("T", ["'(' E ')'", "V", "'C'"], [0.2, 0.55, 0.25])
        grammarstr += construct_production("V", batch_settings["variables"], batch_settings["p_vars"])
        grammar = pg.GeneratorGrammar(grammarstr)

    elif batch_settings["grammar"] == "polystricter":
        grammarstr = construct_production("P", ["P '+' M", "P '-' M", "M"], [0.3, 0.3, 0.4])
        grammarstr += construct_production("M", ["T", "'C' '*' T"], [0.5, 0.5])
        grammarstr += construct_production("T", ["V '*' V", "V"], [0.2, 0.8])
        grammarstr += construct_production("V", batch_settings["variables"], batch_settings["p_vars"])
        grammar = pg.GeneratorGrammar(grammarstr)

    elif batch_settings["grammar"] == "phaseosc":
        grammarstr = construct_production("P", ["P '+' 'C' '*' 'sin''(' A ')'", "'C' '*' 'sin''(' A ')'", "'C'"], [0.4, 0.5, 0.1])
        grammarstr += construct_production("A", ["A '+' T", "A '-' T", "T"], [0.15, 0.15, 0.7])
        grammarstr += construct_production("T", ["V", "'C' '*' V", "'C'"], [0.6, 0.3, 0.1])
        grammarstr += construct_production("V", batch_settings["variables"], batch_settings["p_vars"])
        grammar = pg.GeneratorGrammar(grammarstr)

    elif batch_settings["grammar"] == "phaseosc2":
        grammarstr = construct_production("P", ["P '+' M", "M"], [0.3, 0.7])
        grammarstr += construct_production("M", ["'C'", "'C' '*' K", "K"], [0.1, 0.7, 0.2])
        grammarstr += construct_production("K", ["'sin'" + "'('" + "T" + "')'"], [1.])
        grammarstr += construct_production("T", ["'C' '+' R", "R"], [1/2, 1/2])
        grammarstr += construct_production("R", ["V '+' V", "V"], [1/2, 1/2])
        grammarstr += construct_production("V", batch_settings["variables"], batch_settings["p_vars"])
        grammar = pg.GeneratorGrammar(grammarstr)

    else:
        print("Error: no such grammar.")

    # generate models from grammar
    sym_vars = [batch_settings["variables"][i][1] for i in range(len(batch_settings["variables"]))]
    symbols = {"x": sym_vars, "const": "C"}
    models = generate_models(grammar, symbols, system_size=1,
                             strategy_settings={"N": batch_settings["num_samples"],
                                                "max_repeat": 50})

    model_batches = models.split(n_batches=batch_settings["n_batches"])

    # save batches
    os.makedirs(batch_settings["path_out"], exist_ok=True)

    for ib in range(batch_settings["n_batches"]):
        filename = "structures_{}_{}_{}_n{}_batch{}.pg".format(batch_settings["exp_version"],
                                                               batch_settings["grammar"],
                                                               batch_settings["system_type"],
                                                               batch_settings["num_samples"],
                                                               str(ib))
        with open(batch_settings["path_out"] + filename, "wb") as file:
            pickle.dump(model_batches[ib], file)

    # save info about grammar:
    grammar_info = batch_settings
    grammar_info['grammar_structure'] = grammarstr
    grammar_info_filename = "structures_{}_{}_{}_n{}_grammar_info.txt".format(batch_settings["exp_version"],
                                                                batch_settings["grammar"],
                                                                batch_settings["system_type"],
                                                                batch_settings["num_samples"])

    fo = open(batch_settings["path_out"] + grammar_info_filename, "w")
    for k, v in grammar_info.items():
        fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
    fo.close()

    # do manually
    if batch_settings["manual"]:
        symbols = {"x": sym_vars, "const": "C"}
        models = pg.ModelBox()
        models.add_system(["C*y", "C*y + C*x*x*y + C*x"], symbols=symbols)
        models.add_system(["C*x", "C*y"], symbols=symbols)
        file_name = os.path.join(batch_settings["path_out"], "job_{}_v{}_batchM.pg".format(batch_settings["system_type"], batch_settings["job_version"]))
        with open(file_name, "wb") as file:
            pickle.dump(models, file)

    # create shell file
    if batch_settings["create_sh_file"]:
        create_sh_file(**batch_settings)

if __name__ == '__main__':

    exp_version = "v10"
    path_main = "D:\\Experiments\\MLJ23\\proged\\numdiff"
    path_base_out = f"{path_main}\\structures\\{exp_version}\\"
    n_batches = 20
    #grammars = ["universal", "polynomial", "polybrackets", "polystricter"]
    grammars = ["polynomial"]
    num_samples = 10000
    vars_dict = {
        'onestate': [["'x'", "'y'"], [1/2, 1/2]],
        'lorenz': [["'x'", "'y'", "'z'"], [1/3, 1/3, 1/3]],
        'twophase': [["'x'", "'y'", "'t'"], [2/5, 2/5, 1/5]],
        'twostate1': [["'x'", "'y'", "'u'", "'v'"], [0.35, 0.35, 0.15, 0.15]],
        'twostate2': [["'x'", "'y'", "'u'", "'v'"], [0.15, 0.15, 0.35, 0.35]]
    }

    for igram in grammars:
        print(f"started gram: {igram}")

        for itype in vars_dict.keys():
            # itype = 'twostate'

            # settings
            if (itype == 'twophase') & (igram == "polynomial"):
                igram = 'phaseosc'
            elif (itype == 'twophase') & (igram != "polynomial") & (igram != "universal"):
                continue

            batch_settings = {
                "system_type": itype,
                "exp_version": exp_version,
                "variables": vars_dict[itype][0],
                "grammar": igram,
                "p_vars": vars_dict[itype][1],
                "num_samples": num_samples,
                "n_batches": n_batches,
                "path_out": f"{path_base_out}batchsize{n_batches}\\{itype}\\",
                "manual": False,
                "create_sh_file": False
            }

            print(f"started type: {itype}")
            create_batches(**batch_settings)