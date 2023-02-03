import numpy as np
from ProGED.generators.grammar import GeneratorGrammar

# from ProGED.generators.grammar import GeneratorGrammar

def grammar_from_template(template_name, generator_settings):
    if template_name in GRAMMAR_LIBRARY:
        grammar_str = GRAMMAR_LIBRARY[template_name](**generator_settings)
        return GeneratorGrammar(grammar_str)


def construct_right(right="a", prob=1):
    return right + " [" + str(prob) + "]"


def construct_production(left="S", items=["a"], probs=[1]):
    if not items:
        return ""
    else:
        return "\n" + left + " -> " + construct_right_distribution(items=items, probs=probs)


def construct_right_distribution(items=[], probs=[]):
    p = np.array(probs) / np.sum(probs)
    S = construct_right(right=items[0], prob=p[0])
    for i in range(1, len(items)):
        S += " | " + construct_right(right=items[i], prob=p[i])
    return S


def construct_grammar_trigonometric(probs1=[0.8, 0.2], probs2=[0.4, 0.4, 0.2],
                                    variables=["'x'", "'y'"], p_vars=[0.5, 0.5],
                                    functions=["'sin'", "'cos'", "'tan'"]):
    grammar = construct_production(left="S", items=["T1" + "'('" + "V" + "')'",
                                                    "T1" + " " + "T2" + "'('" + "V" + "')'"], probs=probs1)
    grammar += construct_production(left="T1", items=functions, probs=probs2)
    grammar += construct_production(left="T2", items=["'h'"], probs=[1])
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar


def construct_grammar_function(functions=["'sin'", "'cos'"], probs=[0.5, 0.5], string=True):
    grammar = construct_production(left="S", items=["A'(''x'')'"], probs=[1])
    grammar += construct_production(left="A", items=functions, probs=probs)
    return grammar


def construct_grammar_polytrig(p_more_terms=[0.7, 0.15, 0.15], p_higher_terms=0.5, p_vars=[0.5, 0.3, 0.2],
                               variables=["'x'", "'v'", "'a'", "'sin(C*x + C)'"]):
    grammar = construct_production(left="S", items=["'C' '+' S2"], probs=[1])
    grammar += construct_production(left="S2", items=["'C' '*' T '+' S2", "'C' '*' T", "'C'"], probs=p_more_terms)
    grammar += construct_production(left="T", items=["T '*' V", "V"], probs=[p_higher_terms, 1 - p_higher_terms])
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar


def construct_grammar_polynomial(p_S=[0.4, 0.6], p_T=[0.4, 0.6], p_vars=[1], p_R=[0.6, 0.4], p_F=[1],
                                 functions=["'exp'"], variables=["'x'"]):
    grammar = construct_production(left="S", items=["S '+' R", "R"], probs=p_S)
    grammar += construct_production(left="R", items=["T", "'C' '*' F '(' T ')'"], probs=p_R)
    grammar += construct_production(left="T", items=["T '*' V", "'C'"], probs=p_T)
    grammar += construct_production(left="F", items=functions, probs=p_F)
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar


def construct_grammar_simplerational(p_S=[0.2, 0.8], p_P=[0.4, 0.3, 0.3], p_R=[0.4, 0.6], p_M=[0.4, 0.6],
                                     p_F=[1], p_vars=[1], functions=["'exp'"], variables=["'x'"]):
    grammar = construct_production(left="S", items=["P '/' R", "P"], probs=p_S)
    grammar += construct_production(left="P", items=["P '+' 'C' '*' R", "'C' '*' R", "'C'"], probs=p_P)
    grammar += construct_production(left="R", items=["F '(' 'C' '*' M ')'", "M"], probs=p_R)
    grammar += construct_production(left="M", items=["M '*' V", "V"], probs=p_M)
    grammar += construct_production(left="F", items=functions, probs=p_F)
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar


def construct_grammar_rational(p_S=[0.4, 0.6], p_T=[0.4, 0.6], p_vars=[1], p_R=[0.6, 0.4], p_F=[1],
                               functions=["'exp'"], variables=["'x'"]):
    grammar = construct_production(left="S", items=["'(' E ')' '/' '(' E ')'"], probs=[1])
    grammar += construct_production(left="E", items=["E '+' R", "R"], probs=p_S)
    grammar += construct_production(left="R", items=["T", "'C' '*' F '(' T ')'"], probs=p_R)
    grammar += construct_production(left="T", items=["T '*' V", "'C'"], probs=p_T)
    grammar += construct_production(left="F", items=functions, probs=p_F)
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar


def unit_to_string(unit, unit_symbols=["m", "s", "kg", "T", "V"]):
    return "".join([unit_symbols[i] + str(unit[i]) for i in range(len(unit))])


def string_to_unit(unit_string, unit_symbols=["m", "s", "kg", "T", "V"]):
    u = []
    for i in range(len(unit_symbols) - 1):
        split = unit_string.split(unit_symbols[i])[1].split(unit_symbols[i + 1])
        u += [int(split[0])]
    u += [int(split[1])]
    return u


def units_dict(variables, units, dimensionless=[0, 0, 0, 0, 0], target_variable_unit=[0, 0, 0, 0, 0]):
    dictunits = {}
    for i in range(len(variables)):
        unit_string = unit_to_string(units[i])
        if unit_string in dictunits:
            dictunits[unit_string] += [variables[i]]
        else:
            dictunits[unit_string] = [variables[i]]
    if unit_to_string(dimensionless) not in dictunits:
        dictunits[unit_to_string(dimensionless)] = []
    # if unit_to_string(unit_to_string(units[target_variable_unit_index])) not in dictunits:
    #    dictunits[unit_to_string(units[target_variable_unit_index])] = []
    if unit_to_string(target_variable_unit) not in dictunits:
        dictunits[unit_to_string(target_variable_unit)] = []
    return dictunits


def unit_conversions(units_dict, order=1):
    conversions = {}
    # units = np.array([np.fromstring(unit.strip("[").strip("]").strip(), sep=",", dtype=int) for unit in list(units_dict.keys())])
    units = np.array([string_to_unit(unit) for unit in list(units_dict.keys())])
    for i in range(len(units)):
        conversions_mul = []
        conversions_div = []
        for j in range(len(units)):
            for k in range(len(units)):
                if np.array_equal(units[i], units[j] + units[k]):
                    if [j, k] not in conversions_mul and [k, j] not in conversions_mul:
                        conversions_mul += [[j, k]]
                if np.array_equal(units[i], units[j] - units[k]):
                    if [j, k] not in conversions_div:
                        conversions_div += [[j, k]]
                if np.array_equal(units[i], units[k] - units[j]):
                    if [k, j] not in conversions_div:
                        conversions_div += [[k, j]]
        conversions[str(i) + "*"] = conversions_mul
        conversions[str(i) + "/"] = conversions_div
    return conversions, units


def probs_uniform(items, A=1):
    if len(items) > 0:
        return [A / len(items)] * len(items)
    else:
        return []


def construct_grammar_universal_dim_direct(variables=["'U'", "'d'", "'k'", "'A'"],
                                           p_recursion=[0.1, 0.9],  # recurse vs terminate
                                           p_operations=[0.2, 0.3, 0.4, 0.1],  # sum, sub, mul, div
                                           p_constant=[0.2, 0.8],  # constant vs variable
                                           functions=["sin", "cos", "sqrt", "exp"], p_functs=[0.6, 0.1, 0.1, 0.1, 0.1],
                                           units=[[2, -2, 1, 0, 0], [1, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                                  [2, -2, 1, 0, 0]],
                                           target_variable_unit_index=-1,
                                           dimensionless=[0, 0, 0, 0, 0]):
    target_variable_unit = units[target_variable_unit_index]
    dictunits = units_dict(variables, units)
    conversions, unique_units = unit_conversions(dictunits)
    strunits = [unit_to_string(unit) for unit in unique_units]

    grammar = construct_production(left="S", items=[unit_to_string(target_variable_unit)], probs=[1.0])
    for i in range(len(unique_units)):
        if strunits[i] == unit_to_string(dimensionless):
            grammar += construct_production(left=strunits[i],
                                            items=["F"] + ["'" + f + "(' F ')'" for f in functions],
                                            probs=p_functs)
            left_item = "F"
        else:
            left_item = strunits[i]

        right_sum = ["'('" + strunits[i] + "')'" + "'+'" + "'('" + strunits[i] + "')'"]
        right_sub = ["'('" + strunits[i] + "')'" + "'-'" + "'('" + strunits[i] + "')'"]
        right_mul = ["'('" + strunits[conv[0]] + "')'" + "'*'" + "'('" + strunits[conv[1]] + "')'" for conv in
                     conversions[str(i) + "*"]]
        right_div = ["'('" + strunits[conv[0]] + "')'" + "'/'" + "'('" + strunits[conv[1]] + "')'" for conv in
                     conversions[str(i) + "/"]]
        right_var = dictunits[unit_to_string(unique_units[i])]
        right_const = ["'C'"]
        right_recur = right_sum + right_sub + right_mul + right_div
        right_terminal = right_const + right_var
        right = right_recur + right_terminal

        probs_mul = probs_uniform(right_mul, A=p_operations[2])
        probs_div = probs_uniform(right_div, A=p_operations[3])
        probs_recur = np.hstack([p_operations[:2], probs_mul, probs_div])
        probs_vars = probs_uniform(dictunits[strunits[i]], A=p_constant[1])
        probs_terminal = np.hstack([[p_constant[0]], probs_vars])
        probs = np.hstack([p_recursion[0] * probs_recur, p_recursion[1] * probs_terminal])

        # probs = [0.4/len(right_recur)]*len(right_recur) + [0.6/len(right_terminal)]*len(right_terminal)

        grammar += construct_production(left=left_item,
                                        items=right,
                                        probs=probs)

    return grammar


def construct_grammar_universal_dim(variables=["'U'", "'d'", "'k'"],
                                    p_sum=[0.2, 0.2, 0.6],
                                    p_mul=[0.2, 0.2, 0.6],
                                    p_rec=[0.2, 0.4, 0.4],  # recurse vs terminate
                                    functions=["sin", "cos", "sqrt", "exp"], p_functs=[0.6, 0.1, 0.1, 0.1, 0.1],
                                    units=[[2, -2, 1, 0, 0], [1, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [2, -2, 1, 0, 0]],
                                    target_variable_unit_index=-1,
                                    dimensionless=[0, 0, 0, 0, 0]):
    target_variable_unit = units[target_variable_unit_index]
    dictunits = units_dict(variables, units, dimensionless=dimensionless, target_variable_unit=target_variable_unit)
    conversions, unique_units = unit_conversions(dictunits)
    strunits = [unit_to_string(unit) for unit in unique_units]

    grammar = construct_production(left="S", items=["E_" + unit_to_string(target_variable_unit)], probs=[1.0])
    for i in range(len(unique_units)):
        right_sum = ["E_" + strunits[i] + "'+'" + "F_" + strunits[i]]
        right_sub = ["E_" + strunits[i] + "'-'" + "F_" + strunits[i]]
        right_Fid = ["F_" + strunits[i]]
        grammar += construct_production(left="E_" + strunits[i],
                                        items=right_sum + right_sub + right_Fid,
                                        probs=p_sum)

        right_mul = ["F_" + strunits[conv[0]] + "'*'" + "T_" + strunits[conv[1]] for conv in conversions[str(i) + "*"]]
        right_div = ["F_" + strunits[conv[0]] + "'/'" + "T_" + strunits[conv[1]] for conv in conversions[str(i) + "/"]]
        right_Tid = ["T_" + strunits[i]]
        probs_mul = probs_uniform(right_mul, A=p_mul[0])
        probs_div = probs_uniform(right_div, A=p_mul[1])
        grammar += construct_production(left="F_" + strunits[i],
                                        items=right_mul + right_div + right_Tid,
                                        probs=probs_mul + probs_div + [p_mul[2]])

        if strunits[i] == unit_to_string(dimensionless):
            right_recur = ["F"]
        else:
            right_recur = ["'('" + "E_" + strunits[i] + "')'"]
        right_var = dictunits[unit_to_string(unique_units[i])]
        right_const = ["'C'"]
        probs_vars = probs_uniform(dictunits[strunits[i]], A=p_rec[1])
        grammar += construct_production(left="T_" + strunits[i],
                                        items=right_recur + right_var + right_const,
                                        probs=[p_rec[0]] + probs_vars + [p_rec[2]])

        if strunits[i] == unit_to_string(dimensionless):
            right_F = ["'('" + "E_" + strunits[i] + "')'"] + ["'" + f + "('" + "E_" + strunits[i] + "')'" for f in
                                                              functions]
            grammar += construct_production(left="F",
                                            items=right_F,
                                            probs=p_functs)

    return grammar


def construct_grammar_universal(p_sum=[0.2, 0.2, 0.6], p_mul=[0.2, 0.2, 0.6], p_rec=[0.2, 0.4, 0.4],
                                variables=["'x'", "'y'"], p_vars=[0.5, 0.5],
                                functions=["sin", "asin", "ln", "tanh", "cos", "sqrt", "exp"],
                                p_functs=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    # grammar = construct_production(left="S", items=["E '+' 'C'"], probs=[1])
    grammar = construct_production(left="S", items=["S '+' F", "S '-' F", "F"], probs=p_sum)
    grammar += construct_production(left="F", items=["F '*' T", "F '/' T", "T"], probs=p_mul)
    grammar += construct_production(left="T", items=["R", "'C'", "V"], probs=p_rec)
    # S is changed with V
    grammar += construct_production(left="R", items=["'(' S ')'"] + ["'" + f + "(' S ')'" for f in functions],
                                    probs=p_functs)
    grammar += construct_production(left="V", items=variables, probs=p_vars)
    return grammar


GRAMMAR_LIBRARY = {
    "universal": construct_grammar_universal,
    "universal_dim": construct_grammar_universal_dim,
    "rational": construct_grammar_rational,
    "simplerational": construct_grammar_simplerational,
    "polytrig": construct_grammar_polytrig,
    "trigonometric": construct_grammar_trigonometric,
    "polynomial": construct_grammar_polynomial}


import pandas as pd

def prepare_feynman_dataset(eqfile = "/content/Feynman Equations - Sheet1 - Feynman Equations - Sheet1.csv"):
    reference = pd.read_csv(eqfile)
    dict_equations = {}
    dict_var_names = {}
    dict_var_probs = {}
    for eqN in range(0, 100):
        print(eqN)
        var_number = int(reference["# variables"][eqN])
        print("Number of variables: ")
        print(var_number)
        var_names = ["V_" + reference["v"+str(n)+"_name"][eqN] for n in range(1, var_number + 1)]
        print(var_names)
        output_name = "V_" + reference["Output"][eqN]
        var_probs = [1/var_number]*var_number
        print("Probabilities: ")
        print(var_probs)
        k = str(reference["Manually revised formula"][eqN])
        l = list(k.split(" "))
        print(l)
        dict_equations[k] = l
        dict_var_names[k] = var_names
        dict_var_probs[k] = var_probs
    return dict_equations,dict_var_names,dict_var_probs

#print(l)
print(var_names)
#print(var_probs)

import pandas as pd
from collections import Counter
from nltk import PCFG
from nltk.parse import ViterbiParser
from nltk.parse import ChartParser
from nltk.parse import InsideChartParser
from nltk.parse import BottomUpChartParser
from nltk.parse import pchart
import nltk
import random
from nltk import Tree


def parsing_grammars(grammar, dict_equations, dict_var_names, dict_var_probs):
    grammars = []
    grammars_updated = []
    prod = []
    eqN = 0
    dictionary = {}
    if grammar in GRAMMAR_LIBRARY:
        print("here")
        construct_grammar = GRAMMAR_LIBRARY[grammar]
        for key in dict_equations:
            print(key)
            print(eqN)
            var_names = dict_var_names[key]
            print(var_names)
            var_probs = dict_var_probs[key]
            print(var_probs)
            grammars += [
                PCFG.fromstring(construct_grammar(variables=["'" + v + "'" for v in var_names], p_vars=var_probs))]
            print(grammars[eqN])
            chart_parser = InsideChartParser(grammars[eqN])
            eqN = eqN + 1
            chart_parser.trace(3)
            l = dict_equations[key]
            parses = chart_parser.parse_all(l)
            for parse in chart_parser.parse(l):
                print("Parsed:")
                print(parse)
                print("Productions: /n")
                p = parse.productions()
                print(p)
                print(len(p))
                for m in p:
                    print(m)
                    prod.append(m)

        dictionary = Counter(list(prod))
        print(dictionary)


    else:
        print("The grammar does not exist!")

    return grammars, dictionary

print(dictionary)


def updated_probabilities(grammar, var_names, dictionary):
    grammars_updated = []
    newDict = {}
    eqN = 0
    construct_grammar = GRAMMAR_LIBRARY[grammar]
    if grammar == "universal":
        for key in dictionary:
            value = dictionary[key]
            print(value)
            key = str(key)
            if key == "S -> F" or key == "F -> F '/' T" or key == "T -> R" or key == "T -> 'C'" or key == "T -> V" or key == "S -> S '+' F" or key == "S -> S '-' F" or key == "F -> F '*' T" or key == "F -> T" or key == "R -> '(' S ')'" or key == "R -> 'sin(' S ')'" or key == "R -> 'cos(' S ')'" or key == "R -> 'sqrt(' S ')'" or key == "R -> 'exp(' S ')'" or key == "R -> 'ln(' S ')'" or key == "R -> 'arcsin(' S ')'" or key == "R -> 'tanh(' S ')'":
                newDict[key] = value
        sumT = 0
        sumS = 0
        sumF = 0
        sumR = 0
        dic_sums = {}
        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            if z == "T":
                sumT = sumT + newDict[key]
                print(sumT)
                dic_sums["T"] = sumT
            elif z == "S":
                sumS = sumS + newDict[key]
                print(sumS)
                dic_sums["S"] = sumS
            elif z == "F":
                sumF = sumF + newDict[key]
                print(sumF)
                dic_sums["F"] = sumF
            elif z == "R":
                sumR = sumR + newDict[key]
                print(sumR)
                dic_sums["R"] = sumR

        print(dic_sums)
        prob_sum = list(range(3))
        prob_mul = list(range(3))
        prob_rec = list(range(3))
        prob_func = list(range(8))

        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            val = newDict[key]
            print(val)
            if z == "T":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "T -> R":
                    prob_rec[0] = prob
                elif key == "T -> 'C'":
                    prob_rec[1] = prob
                elif key == "T -> V":
                    prob_rec[2] = prob
            elif z == "S":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "S -> S '+' F":
                    prob_sum[0] = prob
                elif key == "S -> S '-' F":
                    prob_sum[1] = prob
                elif key == "S -> F":
                    prob_sum[2] = prob
            elif z == "F":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "F -> F '*' T":
                    prob_mul[0] = prob
                elif key == "F -> F '/' T":
                    prob_mul[1] = prob
                elif key == "F -> T":
                    prob_mul[2] = prob
            elif z == "R":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "R -> '(' S ')'":
                    prob_func[0] = prob
                elif key == "R -> 'sin(' S ')'":
                    prob_func[1] = prob
                elif key == "R -> 'arcsin(' S ')'":
                    prob_func[2] = prob
                elif key == "R -> 'ln(' S ')'":
                    prob_func[3] = prob
                elif key == "R -> 'tanh(' S ')'":
                    prob_func[4] = prob
                elif key == "R -> 'cos(' S ')'":
                    prob_func[5] = prob
                elif key == "R -> 'sqrt(' S ')'":
                    prob_func[6] = prob
                elif key == "R -> 'exp(' S ')'":
                    prob_func[7] = prob
        print(newDict)
        print("probability sum")
        print(prob_sum)
        for key in var_names:
            # print("vnatre:")
            print(prob_sum)
            var = var_names[key]
            print(var)
            var_number = len(var)
            var_probs = [1 / var_number] * var_number
            grammars_updated += [PCFG.fromstring(
                construct_grammar(variables=["'" + v + "'" for v in var], p_sum=prob_sum, p_mul=prob_mul,
                                  p_rec=prob_rec, p_functs=prob_func, p_vars=var_probs))]
            print(grammars_updated[eqN])
            eqN = eqN + 1
    elif grammar == "rational":
        for key in dictionary:
            value = dictionary[key]
            print(value)
            key = str(key)
            if key == "S -> '(' E ')' '/' '(' E ')'" or key == "E -> E '+' R" or key == "E -> R" or key == "R -> T" or key == "R -> 'C' '*' F '(' T ')'" or key == "T -> T '*' V" or key == "T -> 'C'" or key == "F -> exp":
                newDict[key] = value
        sumS = 0
        sumE = 0
        sumR = 0
        sumT = 0
        sumF = 0
        dic_sums = {}
        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            if z == "T":
                sumT = sumT + newDict[key]
                print(sumT)
                dic_sums["T"] = sumT
            elif z == "S":
                sumS = sumS + newDict[key]
                print(sumS)
                dic_sums["S"] = sumS
            elif z == "F":
                sumF = sumF + newDict[key]
                print(sumF)
                dic_sums["F"] = sumF
            elif z == "R":
                sumR = sumR + newDict[key]
                print(sumR)
                dic_sums["R"] = sumR
            elif z == "E":
                sumE = sumE + newDict[key]
                print(sumE)
                dic_sums["E"] = sumE
        print(dic_sums)
        prob_s = list(range(1))
        prob_e = list(range(2))
        prob_r = list(range(2))
        prob_t = list(range(2))
        prob_f = list(range(1))

        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            val = newDict[key]
            print(val)
            if z == "T":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "T -> T '*' V":
                    prob_t[0] = prob
                elif key == "T -> 'C'":
                    prob_t[1] = prob
            elif z == "S":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "S -> '(' E ')' '/' '(' E ')'":
                    prob_s[0] = prob
            elif z == "F":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "F -> exp":
                    prob_f[0] = prob
            elif z == "R":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "R -> T":
                    prob_r[0] = prob
                elif key == "R -> 'C' '*' F '(' T ')'":
                    prob_r[1] = prob
            elif z == "E":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "E -> E '+' R":
                    prob_e[0] = prob
                elif key == "E -> R":
                    prob_e[1] = prob

        for key in var_names:
            # print("vnatre:")
            print(prob_sum)
            var = var_names[key]
            print(var)
            var_number = len(var)
            var_probs = [1 / var_number] * var_number
            grammars_updated += [PCFG.fromstring(
                construct_grammar(variables=["'" + v + "'" for v in var], p_S=prob_e, p_T=prob_t, p_R=prob_r,
                                  p_F=prob_f, p_vars=var_probs))]
            print(grammars_updated[eqN])
            eqN = eqN + 1

    elif grammar == "simplerational":
        for key in dictionary:
            value = dictionary[key]
            print(value)
            key = str(key)
            if key == "S -> P '/' R" or key == "S -> P" or key == "P -> P '+' 'C' '*' R" or key == "P -> 'C' '*' R" or key == "F -> 'exp'" or key == "P -> 'C'" or key == "R -> F '(' 'C' '*' M ')'" or key == "R -> M" or key == "M -> M '*' V" or key == "M -> V":
                newDict[key] = value
        sumS = 0
        sumP = 0
        sumR = 0
        sumM = 0
        sumF = 0

        dic_sums = {}
        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            if z == "M":
                sumM = sumM + newDict[key]
                print(sumM)
                dic_sums["M"] = sumM
            elif z == "S":
                sumS = sumS + newDict[key]
                print(sumS)
                dic_sums["S"] = sumS
            elif z == "F":
                sumF = sumF + newDict[key]
                print(sumF)
                dic_sums["F"] = sumF
            elif z == "R":
                sumR = sumR + newDict[key]
                print(sumR)
                dic_sums["R"] = sumR
            elif z == "P":
                sumP = sumP + newDict[key]
                print(sumP)
                dic_sums["P"] = sumP

        print(dic_sums)
        prob_s = list(range(2))
        prob_p = list(range(3))
        prob_r = list(range(2))
        prob_m = list(range(2))
        prob_f = list(range(1))

        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            val = newDict[key]
            print(val)
            if z == "S":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "S -> P '/' R":
                    prob_s[0] = prob
                elif key == "S -> P":
                    prob_s[1] = prob
            elif z == "P":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "P -> P '+' 'C' '*' R":
                    prob_p[0] = prob
                elif key == "P -> 'C' '*' R":
                    prob_p[1] = prob
                elif key == "P -> 'C'":
                    prob_p[2] = prob
            elif z == "F":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "F -> exp":
                    prob_f[0] = prob
            elif z == "R":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "R -> F '(' 'C' '*' M ')'":
                    prob_r[0] = prob
                elif key == "R -> M":
                    prob_r[1] = prob
            elif z == "M":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "M -> M '*' V":
                    prob_m[0] = prob
                elif key == "M -> V":
                    prob_m[1] = prob

        for key in var_names:
            # print("vnatre:")
            print(prob_sum)
            var = var_names[key]
            print(var)
            var_number = len(var)
            var_probs = [1 / var_number] * var_number
            grammars_updated += [PCFG.fromstring(
                construct_grammar(variables=["'" + v + "'" for v in var], p_S=prob_s, p_P=probs_p, p_R=probs_r,
                                  p_M=probs_m, p_F=probs_f, p_vars=var_probs))]
            print(grammars_updated[eqN])
            eqN = eqN + 1

    elif grammar == "polytrig":
        for key in dictionary:
            value = dictionary[key]
            print(value)
            key = str(key)
            if key == "S -> 'C' '+' S2" or key == "S2 -> 'C' '*' T '+' S2" or key == "S2 -> 'C' '*' T" or key == "S2 -> 'C'" or key == "T -> T '*' V" or key == "T -> V":
                newDict[key] = value
        sumS = 0
        sumT = 0
        sumS2 = 0
        dic_sums = {}
        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            if z == "S2":
                sumS2 = sumS2 + newDict[key]
                print(sumS2)
                dic_sums["S2"] = sumS2
            elif z == "S":
                sumS = sumS + newDict[key]
                print(sumS)
                dic_sums["S"] = sumS
            elif z == "T":
                sumT = sumT + newDict[key]
                print(sumT)
                dic_sums["T"] = sumT

        print(dic_sums)
        prob_s = list(range(1))
        prob_s2 = list(range(3))
        prob_t = list(range(2))

        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            val = newDict[key]
            print(val)
            if z == "T":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "T -> T '*' V":
                    prob_t[0] = prob
                elif key == "T -> V":
                    prob_t[1] = prob
            elif z == "S":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "S -> 'C' '+' S2":
                    prob_s[0] = prob
            elif z == "S2":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "S2 -> 'C' '*' T '+' S2":
                    prob_s2[0] = prob
                elif key == "S2 -> 'C' '*' T":
                    prob_s2[1] = prob
                elif key == "S2 -> 'C'":
                    prob_s2[2] = prob

        for key in var_names:
            # print("vnatre:")
            print(prob_sum)
            var = var_names[key]
            print(var)
            var_number = len(var)
            var_probs = [1 / var_number] * var_number
            grammars_updated += [PCFG.fromstring(
                construct_grammar(variables=["'" + v + "'" for v in var], p_more_terms=prob_s2, p_higher_terms=prob_t,
                                  p_vars=var_probs))]
            print(grammars_updated[eqN])
            eqN = eqN + 1

    elif grammar == "trigonometric":
        for key in dictionary:
            value = dictionary[key]
            print(value)
            key = str(key)
            if key == "S -> T1 '(' V ')'" or key == "S -> T1 T2 '(' V ')'" or key == "T1 -> 'sin'" or key == "T1 -> 'cos'" or key == "T1 -> 'tan'" or key == "T2 -> 'h'":
                newDict[key] = value
        sumS = 0
        sumT1 = 0
        sumT2 = 0
        dic_sums = {}
        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            if z == "S":
                sumS = sumS + newDict[key]
                print(sumS)
                dic_sums["S"] = sumS
            elif z == "T2":
                sumT2 = sumT2 + newDict[key]
                print(sumT2)
                dic_sums["T2"] = sumT2
            elif z == "T1":
                sumT1 = sumT1 + newDict[key]
                print(sumT1)
                dic_sums["T1"] = sumT1
        print(dic_sums)
        prob_s = list(range(2))
        prob_t1 = list(range(3))
        prob_t2 = list(range(1))

        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            val = newDict[key]
            print(val)
            if z == "T1":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "T1 -> 'sin'":
                    prob_t1[0] = prob
                elif key == "T1 -> 'cos'":
                    prob_t1[1] = prob
                elif key == "T1 -> 'tan'":
                    prob_t1[2] = prob
            elif z == "S":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "S -> T1 '(' V ')'":
                    prob_s[0] = prob
                elif key == "S -> T1 T2 '(' V ')'":
                    prob_s[1] = prob
            elif z == "T2":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "T2 -> 'h'":
                    prob_t2[0] = prob

        for key in var_names:
            # print("vnatre:")
            print(prob_sum)
            var = var_names[key]
            print(var)
            var_number = len(var)
            var_probs = [1 / var_number] * var_number
            grammars_updated += [PCFG.fromstring(
                construct_grammar(variables=["'" + v + "'" for v in var], probs1=prob_s, probs2=prob_t1,
                                  p_vars=var_probs))]
            print(grammars_updated[eqN])
            eqN = eqN + 1

    elif grammar == "polynomial":

        for key in dictionary:
            value = dictionary[key]
            print(value)
            key = str(key)
            if key == "S -> S '+' R" or key == "S -> R" or key == "R -> T" or key == "R -> 'C' '*' F '(' T ')'" or key == "T -> T '*' V" or key == "T -> 'C'" or key == "F -> 'exp'":
                newDict[key] = value
        sumS = 0
        sumR = 0
        sumT = 0
        sumF = 0
        dic_sums = {}
        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            if z == "S":
                sumS = sumS + newDict[key]
                print(sumS)
                dic_sums["S"] = sumS
            elif z == "T":
                sumT = sumT + newDict[key]
                print(sumT)
                dic_sums["T"] = sumT
            elif z == "R":
                sumR = sumR + newDict[key]
                print(sumR)
                dic_sums["R"] = sumR
            elif z == "F":
                sumF = sumF + newDict[key]
                dic_sums["F"] = sumF

        print(dic_sums)
        prob_s = list(range(2))
        prob_t = list(range(2))
        prob_r = list(range(2))
        prob_f = list(range(1))

        for key in newDict:
            m = key.split(" ")
            z = m[0]
            print(z)
            val = newDict[key]
            print(val)
            if z == "T":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "T -> T '*' V":
                    prob_t[0] = prob
                elif key == "T -> 'C'":
                    prob_t[1] = prob
            elif z == "S":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "S -> S '+' R":
                    prob_s[0] = prob
                elif key == "S -> R":
                    prob_s[1] = prob
            elif z == "R":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "R -> T":
                    prob_r[0] = prob
                elif key == "R -> 'C' '*' F '(' T ')'":
                    prob_r[1] = prob
            elif z == "F":
                prob = val / dic_sums[z]
                newDict[key] = prob
                if key == "F -> 'exp'":
                    prob_f[0] = prob

        for key in var_names:
            # print("vnatre:")
            print(prob_sum)
            var = var_names[key]
            print(var)
            var_number = len(var)
            var_probs = [1 / var_number] * var_number
            grammars_updated += [PCFG.fromstring(
                construct_grammar(variables=["'" + v + "'" for v in var], p_S=prob_s, p_R=prob_r, p_T=prob_t,
                                  p_F=prob_f, p_vars=var_probs))]
            print(grammars_updated[eqN])
            eqN = eqN + 1

    return grammars_updated

updated = updated_probabilities("universal",var_names,dictionary)

import pandas as pd
from collections import Counter
from nltk import PCFG
from nltk.parse import ViterbiParser
from nltk.parse import ChartParser
from nltk.parse import InsideChartParser
from nltk.parse import BottomUpChartParser
from nltk.parse import pchart
import nltk
import random
from nltk import Tree


def create_grammars(grammar, eqfile):
    reference = pd.read_csv(eqfile)
    grammars = []
    grammars_updated = []
    prod = []
    for eqN in range(0, 100):
        print(eqN)
        var_number = int(reference["# variables"][eqN])
        print("Number of variables: ")
        print(var_number)
        var_names = ["V_" + reference["v" + str(n) + "_name"][eqN] for n in range(1, var_number + 1)]
        print(var_names)
        output_name = "V_" + reference["Output"][eqN]
        var_probs = [1 / var_number] * var_number
        print("Probabilities: ")
        print(var_probs)
        k = str(reference["Manually revised formula"][eqN])
        l = list(k.split(" "))
        print(l)
        if grammar in GRAMMAR_LIBRARY:
            construct_grammar = GRAMMAR_LIBRARY.get(grammar)
            print(construct_grammar)
            grammars += [
                PCFG.fromstring(construct_grammar(variables=["'" + v + "'" for v in var_names], p_vars=var_probs))]
            print(grammars[eqN])

        rule = []

        chart_parser = InsideChartParser(grammars[eqN])
        chart_parser.trace(3)
        parses = chart_parser.parse_all(l)

        for parse in chart_parser.parse(l):
            print("Parsed:")
            print(parse)
            print("Productions: /n")
            p = parse.productions()
            print(p)
            print(len(p))
            for m in p:
                print(m)
                prod.append(m)

    print(prod)
    dictionary = Counter(list(prod))
    print(dictionary)
    newDict = {}
    for key in dictionary:
        value = dictionary[key]
        print(value)
        key = str(key)
        if key == "S -> F" or key == "F -> F '/' T" or key == "T -> R" or key == "T -> 'C'" or key == "T -> V" or key == "S -> S '+' F" or key == "S -> S '-' F" or key == "F -> F '*' T" or key == "F -> T" or key == "R -> '(' S ')'" or key == "R -> 'sin(' S ')'" or key == "R -> 'cos(' S ')'" or key == "R -> 'sqrt(' S ')'" or key == "R -> 'exp(' S ')'" or key == "R -> 'ln(' S ')'" or key == "R -> 'arcsin(' S ')'" or key == "R -> 'tanh(' S ')'":
            newDict[key] = value

    sumT = 0
    sumS = 0
    sumF = 0
    sumR = 0
    dic_sums = {}
    for key in newDict:
        m = key.split(" ")
        z = m[0]
        print(z)
        if z == "T":
            sumT = sumT + newDict[key]
            print(sumT)
            dic_sums["T"] = sumT
        elif z == "S":
            sumS = sumS + newDict[key]
            print(sumS)
            dic_sums["S"] = sumS
        elif z == "F":
            sumF = sumF + newDict[key]
            print(sumF)
            dic_sums["F"] = sumF
        elif z == "R":
            sumR = sumR + newDict[key]
            print(sumR)
            dic_sums["R"] = sumR

    print(dic_sums)
    prob_sum = list(range(3))
    prob_mul = list(range(3))
    prob_rec = list(range(3))
    prob_func = list(range(8))
    for key in newDict:
        m = key.split(" ")
        z = m[0]
        print(z)
        val = newDict[key]
        print(val)
        if z == "T":
            prob = val / dic_sums[z]
            newDict[key] = prob
            if key == "T -> R":
                prob_rec[0] = prob
            elif key == "T -> 'C'":
                prob_rec[1] = prob
            elif key == "T -> V":
                prob_rec[2] = prob
        elif z == "S":
            prob = val / dic_sums[z]
            newDict[key] = prob
            if key == "S -> S '+' F":
                prob_sum[0] = prob
            elif key == "S -> S '-' F":
                prob_sum[1] = prob
            elif key == "S -> F":
                prob_sum[2] = prob
        elif z == "F":
            prob = val / dic_sums[z]
            newDict[key] = prob
            if key == "F -> F '*' T":
                prob_mul[0] = prob
            elif key == "F -> F '/' T":
                prob_mul[1] = prob
            elif key == "F -> T":
                prob_mul[2] = prob
        elif z == "R":
            prob = val / dic_sums[z]
            newDict[key] = prob
            if key == "R -> '(' S ')'":
                prob_func[0] = prob
            elif key == "R -> 'sin(' S ')'":
                prob_func[1] = prob
            elif key == "R -> 'arcsin(' S ')'":
                prob_func[2] = prob
            elif key == "R -> 'ln(' S ')'":
                prob_func[3] = prob
            elif key == "R -> 'tanh(' S ')'":
                prob_func[4] = prob
            elif key == "R -> 'cos(' S ')'":
                prob_func[5] = prob
            elif key == "R -> 'sqrt(' S ')'":
                prob_func[6] = prob
            elif key == "R -> 'exp(' S ')'":
                prob_func[7] = prob

    print(newDict)
    print(prob_sum)
    for eqN in range(0, 100):
        print(eqN)
        var_number = int(reference["# variables"][eqN])
        var_names = ["V_" + reference["v" + str(n) + "_name"][eqN] for n in range(1, var_number + 1)]
        output_name = "V_" + reference["Output"][eqN]
        var_probs = [1 / var_number] * var_number
        if grammar in GRAMMAR_LIBRARY:
            construct_grammar = GRAMMAR_LIBRARY.get(grammar)
            print(construct_grammar)
            grammars_updated += [PCFG.fromstring(
                construct_grammar(variables=["'" + v + "'" for v in var_names], p_sum=prob_sum, p_mul=prob_mul,
                                  p_rec=prob_rec, p_functs=prob_func, p_vars=var_probs))]
            print(grammars_updated[eqN])
    return grammars_updated


feynman_grammars  = create_grammars("universal",eqfile = "D:\\IJS_MASTER\\ComputationalScientificDiscovery and E-Science\\Feynman Equations - Sheet1.csv")


def prepare_feynman_dataset(
        eqfile="D:\\IJS_MASTER\\ComputationalScientificDiscovery and E-Science\\Feynman Equations - Sheet1.csv"):
    reference = pd.read_csv(eqfile)
    grammars = []
    prod = []
    for eqN in range(0, 100):
        print(eqN)
        var_number = int(reference["# variables"][eqN])
        print("Number of variables: ")
        print(var_number)
        var_names = ["V_" + reference["v" + str(n) + "_name"][eqN] for n in range(1, var_number + 1)]
        print(var_names)
        output_name = "V_" + reference["Output"][eqN]
        var_probs = [1 / var_number] * var_number
        print("Probabilities: ")
        print(var_probs)
        k = str(reference["Manually revised formula"][eqN])
        l = list(k.split(" "))
        print(l)
    return l

