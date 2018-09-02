from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp
import numpy as np
import os,time,random
from schedsets import *

def genrand_vaca(nstaff,vweeks,nprefs):

    init = np.zeros((nstaff,nprefs))

    for s in range(nstaff):
        arr = np.random.choice(vweeks-1,nprefs,replace=False)
        init[s,:] = arr.astype(int)

    return init

    #return np.random.randint(0,high=vweeks-1,size=(nstaff,nprefs))

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

def set_vaca_constraints(solver,v_staff,nstaff,nrots,wk,vacas):

    # Flattened matrix
    v_staff_flat = [v_staff[(rot,0)] for rot in range(nrots)]

    for i in range(nstaff):
        for j in range(vacas.shape[1]):
            val = int(vacas[i,j])
            if val == wk:
                for rot in range(nrots):
                        solver.AddConstraint(v_staff[(rot,0)] != i)

def set_rot_constraints(solver,v_staff,nrots,nwks):

    for i in range(nwks):

        # Can't cover more than 1 rotation
        solver.Add(solver.AllDifferentExcept([v_staff[(j,i)] for j in range(0,ALL_COVR.index('Swing'))],-1))
        solver.Add(solver.AllDifferentExcept([v_staff[(ALL_COVR.index(j),i)] for j in STW_COVR+WMR_COVR+WSP_COVR],-1))

        # BRT
        for j in BRT_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in BRT_STAFF]) == 1)

        # SFL
        for j in SFL_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in SFL_STAFF]) == 1)

        # MSK
        for j in MSK_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)

        # NER
        for j in NER_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in NER_STAFF]) == 1)

        # ABD
        for j in ABD_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in ABD_STAFF]) == 1)

        # CHT
        for j in CHT_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in CHT_STAFF]) == 1)

        # NUC
        for j in NUC_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in NUC_STAFF]) == 1)

        # STA
        for j in STA_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in STA_STAFF]) == 1)

        # OPR
        for j in OPR_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in OPR_STAFF]) == 1)

        # ST3
        for j in ST3_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in ST3_STAFF]) == 1)

        # SWG
        for j in SWG_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in SWG_STAFF]) == 1)

        # STW
        for j in STW_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in STW_STAFF]) == 1)

        # WSP
        for j in WSP_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in WSP_STAFF]) == 1)

        # WMR
        for j in WMR_COVR:
            solver.Add(solver.Max([v_staff[(ALL_COVR.index(j),i)] == ALL_STAFF.index(rad) for rad in WMR_STAFF]) == 1)

def make_random_solver(w):
    # Creates the solver
    solver = pywrapcp.Solver("Schedule Solution")
    random.seed()
    r = int(random.random()*100000)
    print("Week",w,"random seed:", r)
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
        solver.Solve(db,[time_limit_ms,collector])
    else:        
        solver.Solve(db,[collector])

    num_solutions = collector.SolutionCount()
    print("Num solutions",num_solutions)

    return collector

def update_solutions(collect,solutions,v_staff,nrots,week):
    if collect.SolutionCount() > 0:
        bestSolution = collect.SolutionCount() - 1
    else:
        raise ValueError('No solutions found')

    for j in range(nrots):
        solutions[j,week,0] = int(collect.Value(bestSolution,v_staff[(j,0)]))

def get_staffleftover(r,staffVacaSet):

    nleft = -1

    if ALL_COVR[r] in BRT_COVR:
        nleft = len(set(BRT_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in SFL_COVR:
        nleft = len(set(SFL_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in MSK_COVR:
        nleft = len(set(MSK_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in NER_COVR:
        nleft = len(set(NER_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in ABD_COVR:
        nleft = len(set(ABD_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in CHT_COVR:
        nleft = len(set(CHT_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in NUC_COVR:
        nleft = len(set(NUC_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in STA_COVR:
        nleft = len(set(STA_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in OPR_COVR:
        nleft = len(set(OPR_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in ST3_COVR:
        nleft = len(set(ST3_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in SWG_COVR:
        nleft = len(set(SWG_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in STW_COVR:
        nleft = len(set(STW_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in WSP_COVR:
        nleft = len(set(WSP_STAFF).difference(staffVacaSet))
    elif ALL_COVR[r] in WMR_COVR:
        nleft = len(set(WMR_STAFF).difference(staffVacaSet))
    else:
        pass

    return nleft

def update_solutions_working(solutions,vacas,nrots):

    for wk in range(solutions.shape[1]):
        staffVaca = []
        arr = np.where(vacas==wk)
        for i in range(len(arr[0])):
            staffVaca.append(ALL_STAFF[arr[0][i]]) 
        staffVacaSet = set(staffVaca)
        for r in range(nrots):
            solutions[r,wk,1] = get_staffleftover(r,staffVacaSet)

def print_results(solutions):

    for j in range(len(ALL_COVR)):
        line = '{:>20}'.format(ALL_COVR[j])
        for i in range(solutions.shape[1]):
            value = int(solutions[j,i,0])
            left = int(solutions[j,i,1])
            line += '{:>5} {:>3}'.format(ALL_STAFF[value],left)
        print(line)

def print_vacas(v):
    for i in range(v.shape[0]):
        line = '{:>8}'.format(ALL_STAFF[i])
        for j in range(v.shape[1]):
            val = int(v[i,j])
            line += '{:>5}'.format(val)
        print(line)

'''
======
 MAIN
======
'''

def main():

    # Top level parameters
    timeLimit = 10000 # no limit = "0"
    tweeks = 2
    nweeks = 1
    vweeks = 52
    nprefs = 3
    nrots = len(ALL_COVR)
    nstaff = len(ALL_STAFF)

    # Init matrix
    solutions = np.zeros((nrots,tweeks,2))

    # Vacation matrix
    vacas = genrand_vaca(nstaff,vweeks,nprefs)
    print_vacas(vacas)

    for w in range(tweeks):

        # Make solver
        solver = make_random_solver(w)

        # Create variables
        v_staff,v_staff_flat = create_vaca_variables(solver,nstaff,nweeks,nrots)

        # Set constraints based on vacation schedule
        set_vaca_constraints(solver,v_staff,nstaff,nrots,w,vacas)

        # Set rotation constraints
        set_rot_constraints(solver,v_staff,nrots,nweeks)

        # Build multiphase call schedule
        collector = get_collector(solver,v_staff_flat,timeLimit)

        # Update solution matrix
        update_solutions(collector,solutions,v_staff,nrots,w)

    # Update staff left
    update_solutions_working(solutions,vacas,nrots)

    # Print results
    print_results(solutions)
    print("Time limit:",timeLimit)

if __name__ == "__main__":
  main()
