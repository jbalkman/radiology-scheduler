from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp
import numpy as np
import os,time,random
from schedsets import *

def genrand_vaca(nstaff,nweeks):

    return np.random.randint(0,high=51, size=(nstaff,nweeks))

def genrand_bias(nstaff):

    return np.random.randint(0,high=8, size=(nstaff), dtype='int')

def create_vaca_variables(solver,nstaff,nwks,nrots):
    v_staff = {}

    # the staff matrix returns staff for a given slot and given shift
    for i in range(nwks):
        for j in range(nrots):
            v_staff[(j,i)] = solver.IntVar(-1, nstaff-1, "staff(%i,%i)" % (j,i)) # -1 is an escape where shift to not applicable to time of day

    # flattened versions 
    v_staff_flat = [v_staff[(j,i)] for j in range(nrots) for i in range(nwks)]

    return v_staff,v_staff_flat

def set_vaca_constraints(solver,v_staff,nrots,nwks,vacas):

    # Flattened matrix
    v_staff_flat = [v_staff[(rot,wk)] for rot in range(nrots) for wk in range(nwks)]

    for i in range(vacas.shape[0]):
        for j in range(vacas.shape[1]):
            for r in range(nrots):
                solver.AddConstraint(v_staff[(r,vacas[i,j])] != i)

def set_rot_constratints(solver,v_staff,nrots,nwks):

    for i in range(nwks):

        # Can't cover more than 1 rotation
        solver.Add(solver.AllDifferentExcept([v_staff[(j,i)] for j in range(0,ALL_COVR.index('Swing'))],-1))
        solver.Add(solver.AllDifferentExcept([v_staff[(ALL_COVR.index[j],i)] for j in STW_COVR+WMR_COVR+WMR_COVR],-1))

        # BRT
        for j in BRT_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in BRT_STAFF]) == 1)

        # SFL
        for j in SFL_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in SFL_STAFF]) == 1)

        # MSK
        for j in MSK_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)

        # NER
        for j in NER_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in NER_STAFF]) == 1)

        # ABD
        for j in ABD_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in ABD_STAFF]) == 1)

        # CHT
        for j in CHT_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in CHT_STAFF]) == 1)

        # NUC
        for j in NUC_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in NUC_STAFF]) == 1)

        # STA
        for j in STA_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in STA_STAFF]) == 1)

        # OPR
        for j in OPR_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in OPR_STAFF]) == 1)

        # ST3
        for j in ST3_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index[j],i)] == ALL_STAFF.index(rad) for rad in ST3_STAFF]) == 1)

def make_random_solver():
    # Creates the solver
    solver = pywrapcp.Solver("Schedule Solution")
    random.seed()
    r = int(random.random()*100000)
    print("random seed:", r)
    solver.ReSeed(r)

    return solver

def get_collector(solver,v_staff_flat,tlimit):

    # Create the decision builder.
    db = solver.Phase(v_staff_flat, solver.CHOOSE_RANDOM, solver.ASSIGN_RANDOM_VALUE)

    # Create the solution collector.
    solution = solver.Assignment()
    solution.Add(v_staff_flat)

    # Create collector
    #collector = solver.AllSolutionCollector(solution)
    collector = solver.LastSolutionCollector(solution)

    if tlimit > 0:
        time_limit_ms = solver.TimeLimit(tlimit)
        solver.Solve(db,[time_limit_ms,objective,collector])
    else:        
        solver.Solve(db,[collector])

    num_solutions = collector.SolutionCount()
    print("Num solutions",num_solutions)

    return collector

def print_results(collect):

    if collect.SolutionCount() > 0:
        bestSolution = collect.SolutionCount() - 1
    else:
        raise ValueError('No solutions found')

    for j in range(nrots):
        line = '{:>20}'.format(ALL_COVR[j])
        for i in range(nwks):
            value = collect.Value(bestSolution,v_staff[(j,i)])
            line += '{:>5}'.format(ALL_STAFF[value])
        print(line)

'''
======
 MAIN
======
'''

def main():

    # Top level parameters
    timeLimit = 0 # no limit = "0"
    nweeks = 4
    nrots = len(ALL_COVR)
    nstaff = len(ALL_STAFF)

    # Vacation matrix
    vacas = genrand_vaca(nstaff,nweeks)

    # Make solver
    solver = make_random_solver()

    # Create variables
    v_staff,v_staff_flat = create_vaca_variables(solver,nstaff,nweeks,nrots)

    # Set constraints based on vacation schedule
    set_vaca_constraints(solver,v_staff,nrots,nweeks,vacas)

    # Set rotation constraints
    set_rot_constraints(solver,v_staff,nrots,nweeks)

    # Build multiphase call schedule
    collector = get_collector(solver,v_staff_flat,time_limit)

    # Print results
    print_results(collector)
    print("Time limit:",time_limit)

if __name__ == "__main__":
  main()
