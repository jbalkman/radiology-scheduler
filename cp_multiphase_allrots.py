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
Decision Builder Strategies
============================

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
BRT_SHIFTS = ['UNC_Diag_AM','UNC_Diag_PM','UNC_Proc_AM','UNC_Proc_PM','FRE_Mamm','SLN_Mamm']
SFL_SHIFTS = ['FRE_Sonoflu_AM','FRE_Sonoflu_PM','SLN_Sonoflu_AM','SLN_Sonoflu_PM']
MSK_SHIFTS = ['MSK_AM','MSK_PM']
NER_SHIFTS = ['Neuro_AM','Neuro_PM'] 
ABD_SHIFTS = ['Abdomen_AM','Abdomen_PM']
CHT_SHIFTS = ['Chest/PET_AM','Chest/PET_PM']
STA_SHIFTS = ['STAT1_AM','STAT1b_PM','STAT2_PM']
OPR_SHIFTS = ['OPPR1_AM','OPPR2_AM','OPPR3_PM','OPPR4_PM']

# Rotations
BRT_ROTS = ['UNC_Diag','UNC_Proc','FRE_Mamm','SLN_Mamm']
SFL_ROTS = ['FRE_Sonoflu','SLN_Sonoflu']
MSK_ROTS = ['MSK']
NER_ROTS = ['Neuro']
ABD_ROTS = ['Abdomen']
CHT_ROTS = ['Chest/PET']
STA_ROTS = ['STAT_AM','STAT_PM']
OPR_ROTS = ['OPPR_AM','OPPR_PM']

# Staff Lists
ALL_STAFF = ['JDB','SDE','GHL','DCN','JKS','CCM','GJS','GSR','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH','HG','RV']
BRT_STAFF = ['JDB','SDE','GHL','DCN','JKS']
SFL_STAFF = ALL_STAFF
MSK_STAFF = ['CCM','GJS','GSR','DRL','SJP']
NER_STAFF = ['EEP','GSR','JFK','SMN','SJP']
ABD_STAFF = ['BCL','DSL','HSS','JKL','SH']
CHT_STAFF = ['BCL','GJS','SMN','RV','JKL']
STA_STAFF = ['JDB','SDE','GHL','DCN','JKS','CCM','GJS','GSR','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH']

# General Use
WEEKDAYS = ['MON','TUE','WED','THU','FRI']
WEEK_SHIFTS = ['MON-AM','MON-PM','TUE-AM','TUE-PM','WED-AM','WED-PM','THU-AM','THU-PM','FRI-AM','FRI-PM']

# r = staff; a = leave day(s) list; s = solver; l = leave_days variable, st = staff variable; ns = num_shifts
def leave(r,a,s,st,ns):
    rad = BRT_STAFF.index(r)
    for d in a:
        s.Add(s.Max([st[(k,d*2)] == rad for k in range(ns)]) == 0)
        s.Add(s.Max([st[(k,d*2+1)] == rad for k in range(ns)]) == 0)

def print_results(results,section):
    num_staff,num_rots,staff,rots = get_section_nstaff_nrots_staff_rots(section) 

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

'''
===============
 GET FUNCTIONS
===============
'''

def get_section_nstaff_nrots_staff_rots(sect):

    num_staff = 0
    num_rots = 0
    staff = []
    rots = []

    if sect == 'brt':
        num_staff = len(BRT_STAFF)
        num_rots = len(BRT_ROTS)
        staff = BRT_STAFF
        rots = BRT_ROTS
    elif sect == 'sfl':
        num_staff = len(SFL_STAFF)
        num_rots = len(SFL_ROTS)
        staff = SFL_STAFF
        rots = SFL_ROTS
    elif sect == 'msk':
        num_staff = len(MSK_STAFF)
        num_rots = len(MSK_ROTS)
        staff = MSK_STAFF
        rots = MSK_ROTS
    elif sect == 'ner':
        num_staff = len(NER_STAFF)
        num_rots = len(NER_ROTS)
        staff = NER_STAFF
        rots = NER_ROTS
    elif sect == 'abd':
        num_staff = len(ABD_STAFF)
        num_rots = len(ABD_ROTS)
        staff = ABD_STAFF
        rots = ABD_ROTS
    elif sect == 'cht':
        num_staff = len(CHT_STAFF)
        num_rots = len(CHT_SHIFTS)
        staff = CHT_STAFF
        rots = CHT_ROTS
    elif sect == 'sta':
        num_staff = len(STA_STAFF)
        num_rots = len(STA_ROTS)
        staff = STA_STAFF
        rots = STA_ROTS
    elif sect == 'opr':
        num_staff = len(OPR_STAFF)
        num_rots = len(OPR_ROTS)
        staff = OPR_STAFF
        rots = OPR_ROTS
    else:
        raise ValueError('Unresolved section name in get_section_nstaff_nrots_staff_rots function.')
    
    if (num_staff == 0 or num_rots == 0):
        raise ValueError('Exiting function get_section_nstaff_nrots_staff_rots with num_staff or num_rots == 0',num_staff,num_rots)

    return num_staff,num_rots,staff,rots

def get_section_nstaff_nshifts_nrots_shifts_rots(sect):

    num_staff = 0
    num_shifts = 0
    num_rots = 0
    shifts = []
    rots = []

    if sect == 'brt':
        num_staff = len(BRT_STAFF)
        num_shifts = len(BRT_SHIFTS)
        num_rots = len(BRT_ROTS)
        shifts = BRT_SHIFTS
        rots = BRT_ROTS
    elif sect == 'sfl':
        num_staff = len(SFL_STAFF)
        num_shifts = len(SFL_SHIFTS)
        num_rots = len(SFL_ROTS)
        shifts = SFL_SHIFTS
        rots = SFL_ROTS
    elif sect == 'msk':
        num_staff = len(MSK_STAFF)
        num_shifts = len(MSK_SHIFTS)
        num_rots = len(MSK_ROTS)
        shifts = MSK_SHIFTS
        rots = MSK_ROTS
    elif sect == 'ner':
        num_staff = len(NER_STAFF)
        num_shifts = len(NER_SHIFTS)
        num_rots = len(NER_ROTS)
        shifts = NER_SHIFTS
        rots = NER_ROTS
    elif sect == 'abd':
        num_staff = len(ABD_STAFF)
        num_shifts = len(ABD_SHIFTS)
        num_rots = len(ABD_ROTS)
        shifts = ABD_SHIFTS
        rots = ABD_ROTS
    elif sect == 'cht':
        num_staff = len(CHT_STAFF)
        num_shifts = len(CHT_SHIFTS)
        num_rots = len(CHT_ROTS)
        shifts = CHT_SHIFTS
        rots = CHT_ROTS
    elif sect == 'sta':
        num_staff = len(STA_STAFF)
        num_shifts = len(STA_SHIFTS)
        num_rots = len(STA_ROTS)
        shifts = STA_SHIFTS
        rots = STA_ROTS
    elif sect == 'opr':
        num_staff = len(OPR_STAFF)
        num_shifts = len(OPR_SHIFTS)
        num_rots = len(OPR_ROTS)
        shifts = OPR_SHIFTS
        rots = OPR_ROTS
    else:
        raise ValueError('Unresolved section name in get_section_nstaff_nshifts_nrots_shifts_rots function.')
    
    return num_staff,num_shifts,num_rots,shifts,rots

def get_section_nstaff_nshifts_staff_shifts(sect):
    num_staff = 0
    num_shifts = 0
    staff = []
    shifts = []

    if sect == 'brt':
        num_staff = len(BRT_STAFF)
        num_shifts = len(BRT_SHIFTS)
        shifts = BRT_SHIFTS
        staff = BRT_STAFF
    elif sect == 'sfl':
        num_staff = len(SFL_STAFF)
        num_shifts = len(SFL_SHIFTS)
        shifts = SFL_SHIFTS
        staff = SFL_STAFF
    elif sect == 'msk':
        num_staff = len(MSK_STAFF)
        num_shifts = len(MSK_SHIFTS)
        shifts = MSK_SHIFTS
        staff = MSK_STAFF
    elif sect == 'ner':
        num_staff = len(NER_STAFF)
        num_shifts = len(NER_SHIFTS)
        shifts = NER_SHIFTS
        staff = NER_STAFF
    elif sect == 'abd':
        num_staff = len(ABD_STAFF)
        num_shifts = len(ABD_SHIFTS)
        shifts = ABD_SHIFTS
        staff = ABD_STAFF
    elif sect == 'cht':
        num_staff = len(CHT_STAFF)
        num_shifts = len(CHT_SHIFTS)
        shifts = CHT_SHIFTS
        staff = CHT_STAFF
    elif sect == 'sta':
        num_staff = len(STA_STAFF)
        num_shifts = len(STA_SHIFTS)
        shifts = STA_SHIFTS
        staff = STA_STAFF
    elif sect == 'opr':
        num_staff = len(OPR_STAFF)
        num_shifts = len(OPR_SHIFTS)
        shifts = OPR_SHIFTS
        staff = OPR_STAFF
    else:
        raise ValueError('Unresolved section name in get_section_nstaff_nshifts_staff_shifts function.')
    
    return num_staff,num_shifts,staff,shifts

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

def init_brt_bias():
    return np.zeros((len(BRT_STAFF),len(BRT_ROTS)),dtype='int64') - 2 # here the bias is -2 for all rotations; may need to be less for rotations that are less frequent (e.g. -1 for SLN_Mamm)

def init_sfl_bias():
    return np.zeros((len(SFL_STAFF),len(SFL_ROTS)),dtype='int64') - 2

def init_msk_bias():
    return np.zeros((len(MSK_STAFF),len(MSK_ROTS)),dtype='int64') - 2

def init_ner_bias():
    return np.zeros((len(NER_STAFF),len(NER_ROTS)),dtype='int64') - 2

def init_abd_bias():
    return np.zeros((len(ABD_STAFF),len(ABD_ROTS)),dtype='int64') - 2

def init_cht_bias():
    return np.zeros((len(CHT_STAFF),len(CHT_ROTS)),dtype='int64') - 2

def init_sta_bias():
    return np.zeros((len(STA_STAFF),len(STA_ROTS)),dtype='int64') - 2

def init_opr_bias():
    return np.zeros((len(OPR_STAFF),len(OPR_ROTS)),dtype='int64') - 2

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

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)

    for i in range(num_staff):
        sect_allstaff_idx = ALL_STAFF.index(staff[i])
        for j in range(num_hdays):
            if unavail[sect_allstaff_idx,j]:
                for k in range(num_shifts):
                    slvr.Add(stf[(k,j)] != i)

def set_brt_constraints(s,st): # s = solver

  for i in range(len(WEEK_SHIFTS)):

      # No double coverage
      s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(BRT_SHIFTS))],-1))
      
  for i in range(len(WEEKDAYS)):

      # Constraints binding AM/PM rotations
      s.Add(st[(BRT_SHIFTS.index('UNC_Diag_AM'),i*2)] == st[(BRT_SHIFTS.index('UNC_Diag_PM'),i*2+1)])
      s.Add(st[(BRT_SHIFTS.index('UNC_Proc_AM'),i*2)] == st[(BRT_SHIFTS.index('UNC_Proc_PM'),i*2+1)])
      
      # Shifts that don't fit into context (e.g. UNC_Diag_PM on a morning shift)
      s.Add(st[(BRT_SHIFTS.index('UNC_Diag_PM'),i*2)] == -1)
      s.Add(st[(BRT_SHIFTS.index('UNC_Proc_PM'),i*2)] == -1)
      s.Add(st[(BRT_SHIFTS.index('UNC_Diag_AM'),i*2+1)] == -1)
      s.Add(st[(BRT_SHIFTS.index('UNC_Proc_AM'),i*2+1)] == -1)

      s.Add(st[(BRT_SHIFTS.index('UNC_Diag_AM'),i*2)] != -1)
      s.Add(st[(BRT_SHIFTS.index('UNC_Proc_AM'),i*2)] != -1)
      s.Add(st[(BRT_SHIFTS.index('UNC_Diag_PM'),i*2+1)] != -1)
      s.Add(st[(BRT_SHIFTS.index('UNC_Proc_PM'),i*2+1)] != -1)

      # Don't be on the same UNC rotation two days in a row (can relax if short-staffed)
      if i < 4:
          s.Add(st[(BRT_SHIFTS.index('UNC_Proc_AM'),i*2)] != st[(BRT_SHIFTS.index('UNC_Proc_AM'),i*2+2)])
          s.Add(st[(BRT_SHIFTS.index('UNC_Diag_AM'),i*2)] != st[(BRT_SHIFTS.index('UNC_Diag_AM'),i*2+2)])

  # Blocked Schedules (not all rotations are offered on every shift)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),0)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),1)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),2)] != -1)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),3)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),4)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),5)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),6)] != -1)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),7)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),8)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SLN_Mamm'),9)] == -1)
  
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),0)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),1)] == -1)
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),2)] == -1)
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),3)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),4)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),5)] == -1)
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),6)] == -1)
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),7)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),8)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FRE_Mamm'),9)] == -1)

def set_sfl_constraints(s,st): # s = solver
    
    # Don't cover the same Sonoflu shift in 1 week
    s.Add(s.AllDifferent([st[(j*2,i*2)] for j in range(len(SFL_SHIFTS)/2) for i in range(len(WEEKDAYS))]))

    for i in range(len(WEEK_SHIFTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(SFL_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(SFL_SHIFTS.index('FRE_Sonoflu_AM'),i*2)] == st[(SFL_SHIFTS.index('FRE_Sonoflu_PM'),i*2+1)])
        s.Add(st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),i*2)] == st[(SFL_SHIFTS.index('SLN_Sonoflu_PM'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(SFL_SHIFTS.index('FRE_Sonoflu_AM'),i*2)] != -1)
        s.Add(st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),i*2)] != -1)
        s.Add(st[(SFL_SHIFTS.index('FRE_Sonoflu_PM'),i*2+1)] != -1)
        s.Add(st[(SFL_SHIFTS.index('SLN_Sonoflu_PM'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. FRE_Sonoflu_PM on a morning shift)
        s.Add(st[(SFL_SHIFTS.index('FRE_Sonoflu_PM'),i*2)] == -1)
        s.Add(st[(SFL_SHIFTS.index('SLN_Sonoflu_PM'),i*2)] == -1)
        s.Add(st[(SFL_SHIFTS.index('FRE_Sonoflu_AM'),i*2+1)] == -1)
        s.Add(st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),i*2+1)] == -1)

        # Don't be on Sonoflu 2 days in a row
        if i < 4:
            # for same location
            s.Add(st[(SFL_SHIFTS.index('FRE_Sonoflu_AM'),i*2)] != st[(SFL_SHIFTS.index('FRE_Sonoflu_AM'),i*2+2)])
            s.Add(st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),i*2)] != st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),i*2+2)])
            
            # for different location
            s.Add(st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),i*2)] != st[(SFL_SHIFTS.index('FRE_Sonoflu_AM'),i*2+2)])
            s.Add(st[(SFL_SHIFTS.index('FRE_Sonoflu_AM'),i*2)] != st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),i*2+2)])

    # Only MSK person can cover SLN TUE/THU
    s.Add(s.Max([st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),WEEK_SHIFTS.index('TUE-AM'))] == SFL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)
    s.Add(s.Max([st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),WEEK_SHIFTS.index('THU-AM'))] == SFL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)

def set_msk_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SHIFTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(MSK_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(MSK_SHIFTS.index('MSK_AM'),i*2)] == st[(MSK_SHIFTS.index('MSK_PM'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(MSK_SHIFTS.index('MSK_AM'),i*2)] != -1)
        s.Add(st[(MSK_SHIFTS.index('MSK_PM'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(MSK_SHIFTS.index('MSK_PM'),i*2)] == -1)
        s.Add(st[(MSK_SHIFTS.index('MSK_AM'),i*2+1)] == -1)

def set_abd_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SHIFTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(ABD_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(ABD_SHIFTS.index('Abdomen_AM'),i*2)] == st[(ABD_SHIFTS.index('Abdomen_PM'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(ABD_SHIFTS.index('Abdomen_AM'),i*2)] != -1)
        s.Add(st[(ABD_SHIFTS.index('Abdomen_PM'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(ABD_SHIFTS.index('Abdomen_PM'),i*2)] == -1)
        s.Add(st[(ABD_SHIFTS.index('Abdomen_AM'),i*2+1)] == -1)

def set_ner_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SHIFTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(NER_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(NER_SHIFTS.index('Neuro_AM'),i*2)] == st[(NER_SHIFTS.index('Neuro_PM'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(NER_SHIFTS.index('Neuro_AM'),i*2)] != -1)
        s.Add(st[(NER_SHIFTS.index('Neuro_PM'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(NER_SHIFTS.index('Neuro_PM'),i*2)] == -1)
        s.Add(st[(NER_SHIFTS.index('Neuro_AM'),i*2+1)] == -1)

def set_cht_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SHIFTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(CHT_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(CHT_SHIFTS.index('Chest/PET_AM'),i*2)] == st[(CHT_SHIFTS.index('Chest/PET_PM'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(CHT_SHIFTS.index('Chest/PET_AM'),i*2)] != -1)
        s.Add(st[(CHT_SHIFTS.index('Chest/PET_PM'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(CHT_SHIFTS.index('Chest/PET_PM'),i*2)] == -1)
        s.Add(st[(CHT_SHIFTS.index('Chest/PET_AM'),i*2+1)] == -1)

def set_sta_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SHIFTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(STA_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(STA_SHIFTS.index('STAT1_AM'),i*2)] == st[(STA_SHIFTS.index('STAT1b_PM'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(STA_SHIFTS.index('STAT1_AM'),i*2)] != -1)
        s.Add(st[(STA_SHIFTS.index('STAT1b_PM'),i*2+1)] != -1)
        s.Add(st[(STA_SHIFTS.index('STAT2_PM'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(STA_SHIFTS.index('STAT1b_PM'),i*2)] == -1)
        s.Add(st[(STA_SHIFTS.index('STAT2_PM'),i*2)] == -1)
        s.Add(st[(STA_SHIFTS.index('STAT1_AM'),i*2+1)] == -1)

        # Don't be on all day STAT two days in a row 
        if i < 4:
            s.Add(st[(STA_SHIFTS.index('STAT1_AM'),i*2)] != st[(STA_SHIFTS.index('STAT1_AM'),i*2+2)])

def set_opr_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SHIFTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(OPR_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # These shifts are real and need to be assigned
        s.Add(st[(OPR_SHIFTS.index('OPPR1_AM'),i*2)] != -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR2_AM'),i*2)] != -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR3_PM'),i*2+1)] != -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR4_PM'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(OPR_SHIFTS.index('OPPR3_PM'),i*2)] == -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR4_PM'),i*2)] == -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR1_AM'),i*2+1)] != -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR2_AM'),i*2+1)] != -1)
          
'''
====================
 ANALYSIS FUNCTIONS
====================
'''

def create_analysis(collect,stafflookup,cuml,hist,bias,sect):
    print("creating analysis...")

    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)

    analysis = []
    num_solutions = collect.SolutionCount()
    print("Number of solutions:",num_solutions)
    for sol in range(num_solutions):
      
        curr = np.zeros((num_staff,num_shifts,num_hdays))
        for i in range(num_hdays):
            for j in range(num_shifts):
              st = collect.Value(sol,stafflookup[(j,i)])
              if st != -1: # if the rotation is covered by staff (not a placeholder halfday)
                  curr[st,j,i] += 1
        if sect == 'brt':
            updated_cuml,hist_plus = make_brt_hx(curr,cuml,hist,bias)
        elif sect == 'sfl':
            updated_cuml,hist_plus = make_sfl_hx(curr,cuml,hist,bias)
        elif sect == 'msk':
            updated_cuml,hist_plus = make_msk_hx(curr,cuml,hist,bias)
        elif sect == 'ner':
            updated_cuml,hist_plus = make_ner_hx(curr,cuml,hist,bias)
        elif sect == 'abd':
            updated_cuml,hist_plus = make_abd_hx(curr,cuml,hist,bias)
        elif sect == 'cht':
            updated_cuml,hist_plus = make_cht_hx(curr,cuml,hist,bias)
        elif sect == 'sta':
            updated_cuml,hist_plus = make_sta_hx(curr,cuml,hist,bias)
        elif sect == 'opr':
            updated_cuml,hist_plus = make_opr_hx(curr,cuml,hist,bias)
        else:
            raise ValueError('Unresolved section in create_anslysis function.')

        # sort by variance of each matrix; 
        analysis.append((sol,np.var(hist_plus),updated_cuml,hist_plus,curr))

    print("sorting analysis of length", len(analysis))
    analysis.sort(key=lambda x:x[1])

    return analysis

def print_analysis(slvr,collect,stafflookup,anal,sect):
    print("printing analysis...")
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)
    
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

def build_brt(unavail,cuml,hist,bias,limit):

    # Breast settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('brt')
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'brt')
    set_brt_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'brt')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'brt')

    return analysis[0][2],analysis[0][3],analysis[0][4]
    
def build_sfl(unavail,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('sfl')
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'sfl')
    set_sfl_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'sfl')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'sfl')

    return analysis[0][2],analysis[0][3],analysis[0][4]

def build_msk(unavail,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('msk')

    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'msk')
    set_msk_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'msk')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'msk')

    return analysis[0][2],analysis[0][3],analysis[0][4]

def build_ner(unavail,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('ner')
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'ner')
    set_ner_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'ner')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'ner')

    return analysis[0][2],analysis[0][3],analysis[0][4]

def build_abd(unavail,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('abd')
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'abd')
    set_abd_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'abd')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'abd')

    return analysis[0][2],analysis[0][3],analysis[0][4]

def build_cht(unavail,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('cht')
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'cht')
    set_cht_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'cht')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'cht')

    return analysis[0][2],analysis[0][3],analysis[0][4]

def build_sta(unavail,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('sta')
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'sta')
    set_sta_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'sta')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'sta')

    return analysis[0][2],analysis[0][3],analysis[0][4]

def build_opr(unavail,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('opr')
    num_hdays = len(WEEK_SHIFTS)
    num_days = num_hdays/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_hdays,num_shifts,num_staff)

    # Constraints
    set_availability_constraints(solver,staff,unavail,'opr')
    set_opr_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'opr')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'opr')

    return analysis[0][2],analysis[0][3],analysis[0][4]

def build_other(unavail,cuml,hist,bias):
    return cuml, hist

def build_multi(nweeks,sects,limit):
    
    ndays = len(WEEKDAYS)

    unavailability = np.zeros((len(ALL_STAFF),len(WEEK_SHIFTS),nweeks),dtype='bool') # unavailability matrix is in the "shifts" context

    for j in range(len(sects)):
        if sects[j] == 'brt':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('brt')            
            bias = init_brt_bias()
        elif sects[j] == 'sfl':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('sfl')      
            bias = init_sfl_bias()
        elif sects[j] == 'msk':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('msk')      
            bias = init_msk_bias()
        elif sects[j] == 'ner':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('ner')      
            bias = init_ner_bias()
        elif sects[j] == 'abd':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('abd')      
            bias = init_abd_bias()
        elif sects[j] == 'cht':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('cht')      
            bias = init_cht_bias()
            print("CHT staff",nstaff,"CHT nrots",nrots) 
        elif sects[j] == 'sta':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('sta')      
            bias = init_sta_bias()
        elif sects[j] == 'opr':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('opr')      
            bias = init_opr_bias()
        else:
            nstaff = len(OTHER_STAFF)
            nrots = len(OTHER_ROTS)
            bias = init_other_bias()
            
        # cumulative and history are in the "rotation" context
        cumulative = np.zeros((nstaff,nrots,ndays),dtype='int64') 
        history = np.zeros((nstaff,nrots),dtype='int64')

        for i in range(nweeks):
            print("===========================================")
            print("          WEEK #",int(i+1)," ",sects[j])
            print("===========================================")
            
            if sects[j] == 'brt':      
                cumulative,history,recentweek = build_brt(unavailability[:,:,i],cumulative,history,bias,limit) # recentweek is to update_availability matrix
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'brt')
            elif sects[j] == 'sfl':
                cumulative,history,recentweek = build_sfl(unavailability[:,:,i],cumulative,history,bias,limit)
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'sfl')
            elif sects[j] == 'msk':
                cumulative,history,recentweek = build_msk(unavailability[:,:,i],cumulative,history,bias,limit)
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'msk')
            elif sects[j] == 'ner':
                cumulative,history,recentweek = build_ner(unavailability[:,:,i],cumulative,history,bias,limit)
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'ner')
            elif sects[j] == 'abd':
                cumulative,history,recentweek = build_abd(unavailability[:,:,i],cumulative,history,bias,limit)
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'abd')
            elif sects[j] == 'cht':
                cumulative,history,recentweek = build_cht(unavailability[:,:,i],cumulative,history,bias,limit)
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'cht')
            elif sects[j] == 'sta':
                cumulative,history,recentweek = build_sta(unavailability[:,:,i],cumulative,history,bias,limit)
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'sta')
            elif sects[j] == 'opr':
                cumulative,history,recentweek = build_opr(unavailability[:,:,i],cumulative,history,bias,limit)
                unavailability[:,:,i] = update_availability(recentweek,unavailability[:,:,i],'opr')
            else:
                currwk_unavail = unavailability[:,:,i]
                cumulative,history = build_other(currwk_unavail,cumulative,history,bias)

            print_results(cumulative,sects[j])
            print_results(history,sects[j])
            print("======================================================================================")

def update_availability(c,a,s): # c = nstaff x nhds reflecting 1-week (10 shift); a = unavailability matrix
    
    if s == 'brt':
        staff = BRT_STAFF 
    elif s == 'sfl':
        staff = SFL_STAFF 
    elif s == 'msk':
        staff = MSK_STAFF 
    elif s == 'ner':
        staff = NER_STAFF 
    elif s == 'abd':
        staff = ABD_STAFF
    elif s == 'cht':
        staff = CHT_STAFF 
    elif s == 'sta':
        staff = STA_STAFF 
    elif s == 'opr':
        staff = OPR_STAFF 
    else:
        raise ValueError('Unresolved section in update_availability function.')

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

def make_brt_hx(cur,cml,his,bis):
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)

    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('brt')
    
    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j < 2 and i%2 == 1: # the UNC-Diag AM/PM are both considered UNC-Diag
                        curr_rots[s,rots.index('UNC_Diag'),int(i/2)] += 1
                    elif j < 4 and i%2 == 1:  # the UNC-Proc AM/PM are both considered UNC-Proc
                        curr_rots[s,rots.index('UNC_Proc'),int(i/2)] += 1
                    elif j == 4: # FRE_Mamm
                        curr_rots[s,rots.index('FRE_Mamm'),int(i/2)] += 1
                    elif j == 5 : # SLN Mamm
                        curr_rots[s,rots.index('SLN_Mamm'),int(i/2)] += 1
                    else:
#raise ValueError('Unresolved shift/halfday combination in make_brt_hx function.',i,j)
                        pass
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_sfl_hx(cur,cml,his,bis):
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('sfl')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('FRE_Sonoflu_AM') and i%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('FRE_Sonoflu'),int(i/2)] += 1
                    elif j == shifts.index('SLN_Sonoflu_AM') and i%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('SLN_Sonoflu'),int(i/2)] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_sfl_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_msk_hx(cur,cml,his,bis):
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('msk')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('MSK_AM') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('MSK'),int(i/2)] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_msk_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_abd_hx(cur,cml,his,bis):
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('abd')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Abdomen_AM') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('Abdomen'),int(i/2)] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_abd_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_ner_hx(cur,cml,his,bis):
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('ner')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Neuro_AM') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('Neuro'),int(i/2)] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_ner_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_cht_hx(cur,cml,his,bis):
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('cht')
    #print("staff",nstaff,"shifts",nshifts,"nrots",nrots,"shifts",shifts,"rots",rots)
    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Chest/PET_AM') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('Chest/PET'),int(i/2)] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_cht_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')
    print("Curr Rots:",curr_rots)
    print("History:",his)
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_sta_hx(cur,cml,his,bis):
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('sta')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('STAT1_AM') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('STAT_AM'),int(i/2)] += 1
                    elif (j == shifts.index('STAT1b_PM') or j == shifts.index('STAT2_PM')) and i%2 == 1:
                        curr_rots[s,rots.index('STAT_PM'),int(i/2)] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_sta_hx function.')
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_opr_hx(cur,cml,his,bis):
    nhds = len(WEEK_SHIFTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('opr')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nhds):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j < shifts.index('OPPR3_PM') and i%2 == 0: # the AM shifts are indexes 0,1 and the PM shifts are indexes 2,3
                        curr_rots[s,rots.index('OPPR_AM'),int(i/2)] += 1
                    elif j > shifts.index('OPPR2_AM') and i%2 == 1: 
                        curr_rots[s,rots.index('OPPR_PM'),int(i/2)] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_opr_hx function.')
                        
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
    num_weeks = 1
    time_limit = 1000
    sections = ['brt','ner','msk','abd','sta','cht','opr','sfl']
    #sections = ['cht']
    #sections = ['sonoflu']

    # Build multiphase schedule
    build_multi(num_weeks,sections,time_limit)

if __name__ == "__main__":
  main()
