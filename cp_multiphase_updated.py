from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp
import numpy as np
from operator import itemgetter
import os,time,random

'''
TODO
====
- improve multiphase print statements for rigorous testing
- experiment with sharing of special days, like UNCProcs (still not equally distributed); hierarchy of var?
- experiment with higher biases and adjust min value; what should the formula be? number of rotations/staff?
- revisit callback functions for decision tree to implement bias selection at that level
'''

'''
DB STRATEGIES
=============

For Var strategies: (IntVarStrategy)

CHOOSE_FIRST_UNBOUND
CHOOSE_RANDOM
CHOOSE_MIN_SIZE_LOWEST_MIN
CHOOSE_MIN_SIZE_HIGHEST_MIN
CHOOSE_MIN_SIZE_LOWEST_MAX
CHOOSE_MIN_SIZE_HIGHEST_MAX
CHOOSE_LOWEST_MIN
CHOOSE_HIGHEST_MAX
CHOOSE_MIN_SIZE
CHOOSE_MAX_SIZE
CHOOSE_MAX_REGRET
CHOOSE_PATH

For value strategies they are: (IntValueStrategy)

ASSIGN_MIN_VALUE
ASSIGN_MAX_VALUE
ASSIGN_RANDOM_VALUE
ASSIGN_CENTER_VALUE
SPLIT_LOWER_HALF
SPLIT_UPPER_HALF
'''

# Shifts
ALL_SHIFTS = ['UNC_Diag_AM','UNC_Diag_PM','UNC_Proc_AM','UNC_Proc_PM','FRE_Mamm','SLN_Mamm','FRE_Sonoflu_AM','FRE_Sonoflu_PM','SLN_Sonoflu_AM','SLN_Sonoflu_PM','OPPR1','OPPR2','OPPR3','OPPR4']
BRST_SHIFTS = ['UNC_Diag_AM','UNC_Diag_PM','UNC_Proc_AM','UNC_Proc_PM','FRE_Mamm','SLN_Mamm']
SONOFLU_SHIFTS = ['FRE_Sonoflu_AM','FRE_Sonoflu_PM','SLN_Sonoflu_AM','SLN_Sonoflu_PM']
OTHER_SHIFTS = ['OPPR1','OPPR2','OPPR3','OPPR4']

# Rotations
BRST_ROTS = ['UNC_Diag','UNC_Proc','FRE_Mamm','SLN_Mamm']
SONOFLU_ROTS = ['FRE_Sonoflu','SLN_Sonoflu']
OTHER_ROTS = ['OPPRAM','OPPRPM']

# Staff Lists
ALL_STAFF = ['Balkman','Kaufman','Lin','Nwamuo','Sharkey','Edwards','Palrecha','Sabo','Lim','Nayak','Sriram']
BRST_STAFF = ['Balkman','Lin','Nwamuo','Sharkey','Edwards']
SONOFLU_STAFF = ['Balkman','Kaufman','Lin','Nwamuo','Sharkey','Edwards','Palrecha','Sabo','Lim','Nayak','Sriram']
MSK_STAFF = ['Sabo','Lim','Palrecha']
NEURO_STAFF = ['Nayak','Sriram','Palrecha','Kaufman']
OTHER_STAFF = ['Balkman','Lin','Nwamuo','Sharkey','Edwards','Palrecha']

# General Use
WEEKDAYS = ['MON','TUE','WED','THU','FRI']
WEEK_SHIFTS = ['MON-AM','MON-PM','TUE-AM','TUE-PM','WED-AM','WED-PM','THU-AM','THU-PM','FRI-AM','FRI-PM']

# r = staff; a = leave day(s) list; s = solver; l = leave_days variable, st = staff variable; ns = num_shifts
def leave(r,a,s,st,ns):
    rad = BRST_STAFF.index(r)
    for d in a:
        s.Add(s.Max([st[(k,d*2)] == rad for k in range(ns)]) == 0)
        s.Add(s.Max([st[(k,d*2+1)] == rad for k in range(ns)]) == 0)

def print_results(results,section):
    if section == 'breast':
        num_staff = len(BRST_STAFF)
        staff = BRST_STAFF
        num_rots = len(BRST_ROTS)
        rots = BRST_ROTS
    else:
        num_staff = len(SONOFLU_STAFF)
        staff = SONOFLU_STAFF
        num_rots = len(SONOFLU_ROTS)
        rots = SONOFLU_ROTS

    if len(results.shape) > 2:
        for s in range(num_staff):
            print()
            print("Staff",staff[s])
            for r in range(num_rots):
                mon = results[s][r][WEEKDAYS.index('MON')]
                tue = results[s][r][WEEKDAYS.index('TUE')]
                wed = results[s][r][WEEKDAYS.index('WED')]
                thu = results[s][r][WEEKDAYS.index('THU')]
                fri = results[s][r][WEEKDAYS.index('FRI')]
                alwk = mon+tue+wed+thu+fri
                print(rots[r],int(alwk)," MON",int(mon),"TUE",int(tue),"WED",int(wed),"THU",int(thu),"FRI",int(fri))
    else:
        for s in range(num_staff):
            print()
            print("Staff",staff[s])
            for r in range(num_rots):
                alwk = results[s][r]
                print(rots[r],int(alwk))

def create_staff_lookup(solver,num_hdays,num_shifts,num_staff):
    staff = {}
    
    for i in range(num_hdays):
        for j in range(num_shifts):
            staff[(j,i)] = solver.IntVar(-1, num_staff - 1, "staff(%i,%i)" % (j, i)) # -1 is an escape where shift to not applicable to time of day
    staff_flat = [staff[(j, i)] for j in range(num_shifts) for i in range(num_hdays)]

    return staff, staff_flat


def get_collector(slvr,flat,limit):

    # Create the decision builder.
    print("creating decision builder...")
    time_limit_ms = slvr.TimeLimit(limit)
    db = slvr.Phase(flat, slvr.CHOOSE_RANDOM, slvr.ASSIGN_RANDOM_VALUE)

    # Create the solution collector.
    print("creating collector...")
    solution = slvr.Assignment()
    solution.Add(flat)
    collector = slvr.AllSolutionCollector(solution)
    slvr.Solve(db,[time_limit_ms, collector])
    return collector

'''
================
 BIAS FUNCTIONS
================
'''

def init_brst_bias():
    return np.zeros((len(BRST_STAFF),len(BRST_ROTS)),dtype='int64') - 2 # here the bias is -2 for all rotations; may need to be less for rotations that are less frequent (e.g. -1 for SLN_Mamm)

def init_sonoflu_bias():
    return np.zeros((len(SONOFLU_STAFF),len(SONOFLU_ROTS)),dtype='int64') - 2 # here the bias is -2 for all rotations; may need to be less for rotations that are less frequent (e.g. -1 for SLN_Mamm)

def init_other_bias():
    pass

def add_history_logic(old,curr):
    minimum = -10 # establish a saturation point to prevent runaway values

    if old < 0 and curr > 0:
        return 1
    elif curr > 0:
        return (old+1)
    elif old == minimum-1:
        return minimum
    else:
        return old

add_history_matrix = np.vectorize(add_history_logic)

'''
======================
 CONSTRAINT FUNCTIONS
======================
'''

def set_availability_constraints(slvr,stf,unavail,sect):

    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2

    if sect == 'breast':
        num_staff = len(BRST_STAFF)
        num_shifts = len(BRST_SHIFTS)
        shifts = BRST_SHIFTS
        staff = BRST_STAFF

    elif sect == 'sonoflu':
        num_staff = len(SONOFLU_STAFF)
        num_shifts = len(SONOFLU_SHIFTS)
        shifts = SONOFLU_SHIFTS
        staff = SONOFLU_STAFF
    else:
        pass

    for i in range(num_staff):
        sect_allstaff_idx = ALL_STAFF.index(staff[i])
        for j in range(num_hdays):
            if unavail[sect_allstaff_idx,j]:
                for k in range(num_shifts):
                    slvr.Add(stf[(k,j)] != i)

def set_brst_constraints(s,st): # s = solver

  for i in range(len(WEEK_SHIFTS)):

      # No double coverage
      s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(BRST_SHIFTS))],-1))
      
  for i in range(len(WEEKDAYS)):

      # Constraints binding AM/PM rotations
      s.Add(st[(BRST_SHIFTS.index('UNC_Diag_AM'),i*2)] == st[(BRST_SHIFTS.index('UNC_Diag_PM'),i*2+1)])
      s.Add(st[(BRST_SHIFTS.index('UNC_Proc_AM'),i*2)] == st[(BRST_SHIFTS.index('UNC_Proc_PM'),i*2+1)])
      
      # Shifts that don't fit into context (e.g. UNC_Diag_PM on a morning shift)
      s.Add(st[(BRST_SHIFTS.index('UNC_Diag_PM'),i*2)] == -1)
      s.Add(st[(BRST_SHIFTS.index('UNC_Proc_PM'),i*2)] == -1)
      s.Add(st[(BRST_SHIFTS.index('UNC_Diag_AM'),i*2+1)] == -1)
      s.Add(st[(BRST_SHIFTS.index('UNC_Proc_AM'),i*2+1)] == -1)

      s.Add(st[(BRST_SHIFTS.index('UNC_Diag_AM'),i*2)] != -1)
      s.Add(st[(BRST_SHIFTS.index('UNC_Proc_AM'),i*2)] != -1)
      s.Add(st[(BRST_SHIFTS.index('UNC_Diag_PM'),i*2+1)] != -1)
      s.Add(st[(BRST_SHIFTS.index('UNC_Proc_PM'),i*2+1)] != -1)

      # Don't be on the same UNC rotation two days in a row (can relax if short-staffed)
      if i < 4:
          s.Add(st[(BRST_SHIFTS.index('UNC_Proc_AM'),i*2)] != st[(BRST_SHIFTS.index('UNC_Proc_AM'),i*2+2)])
          s.Add(st[(BRST_SHIFTS.index('UNC_Diag_AM'),i*2)] != st[(BRST_SHIFTS.index('UNC_Diag_AM'),i*2+2)])

  # Blocked Schedules (not all rotations are offered on every shift)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),0)] == -1)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),1)] == -1)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),2)] != -1)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),3)] == -1)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),4)] == -1)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),5)] == -1)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),6)] != -1)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),7)] == -1)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),8)] == -1)
  s.Add(st[(BRST_SHIFTS.index('SLN_Mamm'),9)] == -1)
  
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),0)] != -1)
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),1)] == -1)
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),2)] == -1)
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),3)] != -1)
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),4)] != -1)
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),5)] == -1)
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),6)] == -1)
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),7)] != -1)
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),8)] != -1)
  s.Add(st[(BRST_SHIFTS.index('FRE_Mamm'),9)] == -1)

def set_sonoflu_constraints(s,st): # s = solver
    
    # Don't cover the same Sonoflu shift in 1 week
    s.Add(s.AllDifferent([st[(j*2,i*2)] for j in range(len(SONOFLU_SHIFTS)/2) for i in range(len(WEEKDAYS))]))

    for i in range(len(WEEK_SHIFTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(SONOFLU_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_AM'),i*2)] == st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_PM'),i*2+1)])
        s.Add(st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_AM'),i*2)] == st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_PM'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_AM'),i*2)] != -1)
        s.Add(st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_AM'),i*2)] != -1)
        s.Add(st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_PM'),i*2+1)] != -1)
        s.Add(st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_PM'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. FRE_Sonoflu_PM on a morning shift)
        s.Add(st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_PM'),i*2)] == -1)
        s.Add(st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_PM'),i*2)] == -1)
        s.Add(st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_AM'),i*2+1)] == -1)
        s.Add(st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_AM'),i*2+1)] == -1)

        # Don't be on Sonoflu 2 days in a row
        if i < 4:
            # for same location
            s.Add(st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_AM'),i*2)] != st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_AM'),i*2+2)])
            s.Add(st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_AM'),i*2)] != st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_AM'),i*2+2)])
            
            # for different location
            s.Add(st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_AM'),i*2)] != st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_AM'),i*2+2)])
            s.Add(st[(SONOFLU_SHIFTS.index('FRE_Sonoflu_AM'),i*2)] != st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_AM'),i*2+2)])

    # Only MSK person can cover SLN TUE/THU
    s.Add(s.Max([st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_AM'),WEEK_SHIFTS.index('TUE-AM'))] == SONOFLU_STAFF.index(rad) for rad in MSK_STAFF]) == 1)
    s.Add(s.Max([st[(SONOFLU_SHIFTS.index('SLN_Sonoflu_AM'),WEEK_SHIFTS.index('THU-AM'))] == SONOFLU_STAFF.index(rad) for rad in MSK_STAFF]) == 1)

'''
====================
 ANALYSIS FUNCTIONS
====================
'''

def create_analysis(collect,stafflookup,cuml,hist,bias,sect):
    print("creating analysis...")

    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2

    if sect == 'breast':
        num_staff = len(BRST_STAFF)
        num_shifts = len(BRST_SHIFTS)
        shifts = BRST_SHIFTS
        staff = BRST_STAFF
    elif sect == 'sonoflu':
        num_staff = len(SONOFLU_STAFF)
        num_shifts = len(SONOFLU_SHIFTS)
        shifts = SONOFLU_SHIFTS
        staff = SONOFLU_STAFF
    else:
        pass

    analysis = []

    for sol in range(collect.SolutionCount()):
      
        curr = np.zeros((num_staff,num_shifts,num_hdays))
        for i in range(num_hdays):
            for j in range(num_shifts):
              st = collect.Value(sol,stafflookup[(j,i)])
              if st != -1: # if the rotation is covered by staff (not a placeholder halfday)
                  curr[st,j,i] += 1
        if sect == 'breast':
            updated_cuml,hist_plus = make_brst_hx(curr,cuml,hist,bias)
        elif sect == 'sonoflu':
            updated_cuml,hist_plus = make_sonoflu_hx(curr,cuml,hist,bias)
        else:
          updated_cuml,hist_plus = make_brst_hx(curr,cuml,hist,bias)

        # sort by variance of each matrix; 
        analysis.append((sol,np.var(hist_plus),updated_cuml,hist_plus,curr))

    print("sorting analysis of length", len(analysis))
    analysis.sort(key=lambda x:x[1])

    return analysis

def print_analysis(slvr,collect,stafflookup,anal,sect):
    print("printing analysis...")
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2

    if sect == 'breast':
        num_staff = len(BRST_STAFF)
        num_shifts = len(BRST_SHIFTS)
        shifts = BRST_SHIFTS
        staff = BRST_STAFF
    elif sect == 'sonoflu':
        num_staff = len(SONOFLU_STAFF)
        num_shifts = len(SONOFLU_SHIFTS)
        shifts = SONOFLU_SHIFTS
        staff = SONOFLU_STAFF
    else:
        pass
    
    ts = anal[0][0] # ts = top solution

    print()
    print("Staffing matrix with variance:", anal[0][1])
    for i in range(num_hdays):
        if i%2 == 0:
            print()
            print("Day", i/2)
            for j in range(num_shifts):
                st = collect.Value(ts,stafflookup[(j,i)])
                if st != -1:
                    print("AM Shift:", shifts[j], staff[st])
        else:
            for j in range(num_shifts):
                st = collect.Value(ts,stafflookup[(j,i)])
                if st != -1:
                    print("PM Shift:", shifts[j], staff[st])
    print()
    print("Solutions found:", collect.SolutionCount())
    print("Variance max:", max(anal,key=itemgetter(1))[1])
    print("Variance min:", min(anal,key=itemgetter(1))[1])
    print("Time:", slvr.WallTime(), "ms")
    print()

'''
=================
 BUILD FUNCTIONS
=================
'''

def build_breast(unavail,cuml,hist,bias):

    # Breast settings
    num_staff = len(BRST_STAFF)
    num_shifts = len(BRST_SHIFTS)
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = 500

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'breast')
    set_brst_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'breast')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'breast')

    return analysis[0][2],analysis[0][3],analysis[0][4]
    
def build_sonoflu(unavail,cuml,hist,bias):
    
    # Sonoflu settings
    num_staff = len(SONOFLU_STAFF)
    num_shifts = len(SONOFLU_SHIFTS)
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = 500

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'sonoflu')
    set_sonoflu_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'sonoflu')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'sonoflu')

    return analysis[0][2],analysis[0][3],analysis[0][4]

def build_other(unavail,cuml,hist,bias):
    return cuml, hist

def build_multi(nweeks,sects):
    
    unavailability = np.zeros((len(ALL_STAFF),len(WEEK_SHIFTS),nweeks),dtype='bool') # unavailability matrix is in the "shifts" context

    for j in range(len(sects)):
        if sects[j] == 'breast':
            nstaff = len(BRST_STAFF)
            nrots = len(BRST_ROTS)
            ndays = len(WEEKDAYS)
            
            bias = init_brst_bias()

        elif sects[j] == 'sonoflu':
            nstaff = len(SONOFLU_STAFF)
            nrots = len(SONOFLU_ROTS)
            ndays = len(WEEKDAYS)

            bias = init_sonoflu_bias()

        else:
            nstaff = len(OTHER_STAFF)
            nrots = len(OTHER_ROTS)
            ndays = len(WEEKDAYS)
            
            bias = init_other_bias()
            
        # cumulative and history are in the "rotation" context
        cumulative = np.zeros((nstaff,nrots,ndays),dtype='int64') 
        history = np.zeros((nstaff,nrots),dtype='int64')

        for i in range(nweeks):

            if sects[j] == 'breast':      
                
                cumulative,history,recentweek = build_breast(unavailability[:,:,i],cumulative,history,bias) # recentweek is to update_availability matrix
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'breast')

            elif sects[j] == 'sonoflu':
                
                cumulative,history,recentweek = build_sonoflu(unavailability[:,:,i],cumulative,history,bias)
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'sonoflu')

            else:

                currwk_unavail = unavailability[:,:,i]
                cumulative,history = build_other(currwk_unavail,cumulative,history,bias)

            print_results(cumulative,sects[j])
            print_results(history,sects[j])
            print("======================================================================================")

def update_availability(c,a,s): # c = nstaff x nhds reflecting 1-week (10 shift); a = unavailability matrix
    
    if s == 'breast':
        staff = BRST_STAFF 
    elif s == 'sonoflu':
        staff = SONOFLU_STAFF 
    else:
        staff = OTHER_STAFF

    c_or = np.sum(c,axis=1,dtype='bool')
    for i in range(len(staff)):
        a[ALL_STAFF.index(staff[i]),:] = a[ALL_STAFF.index(staff[i]),:] | c_or[i]

    '''
    # Helpful print debug statements
    print("Availability matrix shape within the update function",a.shape)
    print("Current week matrix shape",c.shape)        
    print("A shape:",a.shape,"C_OR shape:",c_or.shape)
    print("A",a)
    print("C_OR",c_or)
    '''
    
    return a
        
def make_random_solver():
    # Creates the solver
    solver = pywrapcp.Solver("Schedule Solution")
    random.seed()
    r = int(random.random()*100000)
    print("SEED:", r)
    solver.ReSeed(r)

    return solver

def all_staff_idx(s):
    return ALL_STAFF.index(s)

'''
===================
 HISTORY FUNCTIONS
===================
'''

def make_brst_hx(cur,cml,his,bis):
    nstaff = len(BRST_STAFF)
    nshifts = len(BRST_SHIFTS)
    nrots = len(BRST_ROTS)
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)
    
    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j < 2 and i%2 == 1: # the UNC-Diag AM/PM are both considered UNC-Diag
                        curr_rots[s,BRST_ROTS.index('UNC_Diag'),int(i/2)] += 1
                    elif j < 4 and i%2 == 1:  # the UNC-Proc AM/PM are both considered UNC-Proc
                        curr_rots[s,BRST_ROTS.index('UNC_Proc'),int(i/2)] += 1
                    elif j == 4: # FRE_Mamm
                        curr_rots[s,BRST_ROTS.index('FRE_Mamm'),int(i/2)] += 1
                    elif j == 5 : # SLN Mamm
                        curr_rots[s,BRST_ROTS.index('SLN_Mamm'),int(i/2)] += 1
                    else:
                        pass

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_sonoflu_hx(cur,cml,his,bis):
    nstaff = len(SONOFLU_STAFF)
    nshifts = len(SONOFLU_SHIFTS)
    nrots = len(SONOFLU_ROTS)
    shifts = SONOFLU_SHIFTS
    rots = SONOFLU_ROTS
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)
    
    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('FRE_Sonoflu_AM') and i%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,SONOFLU_ROTS.index('FRE_Sonoflu'),int(i/2)] += 1
                    elif j == shifts.index('SLN_Sonoflu_AM') and i%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,SONOFLU_ROTS.index('SLN_Sonoflu'),int(i/2)] += 1
                    else:
                        pass

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

'''
======
 MAIN
======
'''

def main():

    # Top level settings
    num_weeks = 6
    sections = ['breast','sonoflu']
    #sections = ['breast']
    #sections = ['sonoflu']

    # Build multiphase schedule
    build_multi(num_weeks,sections)

if __name__ == "__main__":
  main()
