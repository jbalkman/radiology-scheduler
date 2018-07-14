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

# Legend

# BRT = Breast Imaging
# SFL = Ultrasound/Fluoroscopy
# MSK = Musculoskeletal Imaging
# NER = Neuroradiology
# ABD = Abdominal Imaging
# CHT = Chest/PET Imaging
# STA = STAT/Emergency Imaging
# OPR = Outpatient Plain Film Radiography

# Shifts - to fill schedule
ALL_SHIFTS = ['None','UNC_Diag_AM','UNC_Diag_PM','UNC_Proc_AM','UNC_Proc_PM','FRE_Mamm','SLN_Mamm','FRE_Sonoflu_AM','FRE_Sonoflu_PM','SLN_Sonoflu_AM','SLN_Sonoflu_PM','MSK_AM','MSK_PM','Neuro_AM','Neuro_PM','Abdomen_AM','Abdomen_PM','Chest/PET_AM','Chest/PET_PM','STAT1_AM','STAT1b_PM','STAT2_PM','OPPR1_AM','OPPR2_AM','OPPR3_PM','OPPR4_PM','STAT3','Swing','STATW_AM','STATW_PM','WUSPR','WMR']
BRT_SHIFTS = ['UNC_Diag_AM','UNC_Diag_PM','UNC_Proc_AM','UNC_Proc_PM','FRE_Mamm','SLN_Mamm']
SFL_SHIFTS = ['FRE_Sonoflu_AM','FRE_Sonoflu_PM','SLN_Sonoflu_AM','SLN_Sonoflu_PM']
MSK_SHIFTS = ['MSK_AM','MSK_PM']
NER_SHIFTS = ['Neuro_AM','Neuro_PM'] 
ABD_SHIFTS = ['Abdomen_AM','Abdomen_PM']
CHT_SHIFTS = ['Chest/PET_AM','Chest/PET_PM']
STA_SHIFTS = ['STAT1_AM','STAT1b_PM','STAT2_PM']
OPR_SHIFTS = ['OPPR1_AM','OPPR2_AM','OPPR3_PM','OPPR4_PM']
ST3_SHIFTS = ['STAT3']
SWG_SHIFTS = ['Swing']
STW_SHIFTS = ['STATW_AM','STATW_PM']
WSP_SHIFTS = ['WUSPR']
WMR_SHIFTS = ['WMR']

# Rotations - to measure equality
BRT_ROTS = ['UNC_Diag','UNC_Proc','FRE_Mamm','SLN_Mamm']
SFL_ROTS = ['FRE_Sonoflu','SLN_Sonoflu']
MSK_ROTS = ['MSK']
NER_ROTS = ['Neuro']
ABD_ROTS = ['Abdomen']
CHT_ROTS = ['Chest/PET']
STA_ROTS = ['STAT_AM','STAT_PM']
OPR_ROTS = ['OPPR_AM','OPPR_PM']
ST3_ROTS = ['STAT3']
SWG_ROTS = ['Swing']
STW_ROTS = ['STATW_AM','STATW_PM']
WSP_ROTS = ['WUSPR']
WMR_ROTS = ['WMR']

# Staff Lists
ALL_STAFF = ['JDB','SDE','GHL','DCN','JKS','CCM','GJS','GSR','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH','HG','RV']
BRT_STAFF = ['JDB','SDE','GHL','DCN','JKS']
SFL_STAFF = ALL_STAFF
MSK_STAFF = ['CCM','GJS','GSR','DRL','SJP']
NER_STAFF = ['EEP','GSR','JFK','SMN','SJP']
ABD_STAFF = ['BCL','DSL','HSS','JKL','SH']
CHT_STAFF = ['BCL','GJS','SMN','RV','JKL']
STA_STAFF = ['JDB','SDE','GHL','DCN','JKS','CCM','GJS','GSR','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH']
OPR_STAFF = ALL_STAFF
ST3_STAFF = ['JDB','SDE','GHL','DCN','JKS','GJS','GSR','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH','RV']
SWG_STAFF = ALL_STAFF
STW_STAFF = ST3_STAFF
WSP_STAFF = ['JDB','SDE','GHL','DCN','JKS','BCL','DSL','HSS','JKL','HG','RV']
WMR_STAFF = ['GJS','GSR','DRL','SJP','EEP','JFK','SMN','SH']

# General Use
WEEKDAYS = ['MON','TUE','WED','THU','FRI']
CALLDAYS = ['MON','TUE','WED','THU','FRI','SAT','SUN']
WEEK_SLOTS = ['MON-AM','MON-PM','TUE-AM','TUE-PM','WED-AM','WED-PM','THU-AM','THU-PM','FRI-AM','FRI-PM']
CALL_SLOTS = ['MON-PM','TUE-PM','WED-PM','THU-PM','FRI-PM','SAT-AM','SAT-PM','SUN-AM','SUN-PM']
CALL_SECTS = ['stw','wsp','wmr']

# function currently not used but serves as an example for adding constraints for staff leave/vacation
# r = staff; a = leave day(s) list; s = solver; l = leave_days variable, st = staff variable; ns = num_shifts
def leave(r,a,s,st,ns):
    rad = BRT_STAFF.index(r)
    for d in a:
        s.Add(s.Max([st[(k,d*2)] == rad for k in range(ns)]) == 0)
        s.Add(s.Max([st[(k,d*2+1)] == rad for k in range(ns)]) == 0)

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
        num_rots = len(CHT_ROTS)
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
    elif sect == 'st3':
        num_staff = len(ST3_STAFF)
        num_rots = len(ST3_ROTS)
        staff = ST3_STAFF
        rots = ST3_ROTS
    elif sect == 'swg':
        num_staff = len(SWG_STAFF)
        num_rots = len(SWG_ROTS)
        staff = SWG_STAFF
        rots = SWG_ROTS
    elif sect == 'stw':
        num_staff = len(STW_STAFF)
        num_rots = len(STW_ROTS)
        staff = STW_STAFF
        rots = STW_ROTS
    elif sect == 'wsp':
        num_staff = len(WSP_STAFF)
        num_rots = len(WSP_ROTS)
        staff = WSP_STAFF
        rots = WSP_ROTS
    elif sect == 'wmr':
        num_staff = len(WMR_STAFF)
        num_rots = len(WMR_ROTS)
        staff = WMR_STAFF
        rots = WMR_ROTS
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
    elif sect == 'st3':
        num_staff = len(ST3_STAFF)
        num_shifts = len(ST3_SHIFTS)
        num_rots = len(ST3_ROTS)
        shifts = ST3_SHIFTS
        rots = ST3_ROTS
    elif sect == 'swg':
        num_staff = len(SWG_STAFF)
        num_shifts = len(SWG_SHIFTS)
        num_rots = len(SWG_ROTS)
        shifts = SWG_SHIFTS
        rots = SWG_ROTS
    elif sect == 'stw':
        num_staff = len(STW_STAFF)
        num_shifts = len(STW_SHIFTS)
        num_rots = len(STW_ROTS)
        shifts = STW_SHIFTS
        rots = STW_ROTS
    elif sect == 'wsp':
        num_staff = len(WSP_STAFF)
        num_shifts = len(WSP_SHIFTS)
        num_rots = len(WSP_ROTS)
        shifts = WSP_SHIFTS
        rots = WSP_ROTS
    elif sect == 'wmr':
        num_staff = len(WMR_STAFF)
        num_shifts = len(WMR_SHIFTS)
        num_rots = len(WMR_ROTS)
        shifts = WMR_SHIFTS
        rots = WMR_ROTS
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
    elif sect == 'st3':
        num_staff = len(ST3_STAFF)
        num_shifts = len(ST3_SHIFTS)
        shifts = ST3_SHIFTS
        staff = ST3_STAFF
    elif sect == 'swg':
        num_staff = len(SWG_STAFF)
        num_shifts = len(SWG_SHIFTS)
        shifts = SWG_SHIFTS
        staff = SWG_STAFF
    elif sect == 'stw':
        num_staff = len(STW_STAFF)
        num_shifts = len(STW_SHIFTS)
        shifts = STW_SHIFTS
        staff = STW_STAFF
    elif sect == 'wsp':
        num_staff = len(WSP_STAFF)
        num_shifts = len(WSP_SHIFTS)
        shifts = WSP_SHIFTS
        staff = WSP_STAFF
    elif sect == 'wmr':
        num_staff = len(WMR_STAFF)
        num_shifts = len(WMR_SHIFTS)
        shifts = WMR_SHIFTS
        staff = WMR_STAFF
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

def init_st3_bias():
    return np.zeros((len(ST3_STAFF),len(ST3_ROTS)),dtype='int64') - 1

def init_swg_bias():
    return np.zeros((len(SWG_STAFF),len(SWG_ROTS)),dtype='int64') - 1

def init_stw_bias():
    return np.zeros((len(STW_STAFF),len(STW_ROTS)),dtype='int64') - 1

def init_wsp_bias():
    return np.zeros((len(WSP_STAFF),len(WSP_ROTS)),dtype='int64') - 1

def init_wmr_bias():
    return np.zeros((len(WMR_STAFF),len(WMR_ROTS)),dtype='int64') - 1

def add_history_logic(old,curr):
    '''minimum = -10 # establish a saturation point to prevent runaway values
    if old < 0 and curr > 0:
        return 1
    elif curr > 0:
        return (old+1)
    elif old == minimum-1:
        return minimum
    else:
        return old'''

    if old < 0 and curr > 0:
        #return int(old/2)
        return 2
    elif curr > 0:
        return (old+1)
    else:
        return old

add_history_matrix = np.vectorize(add_history_logic)

'''
======================
 CONSTRAINT FUNCTIONS
======================
'''

def set_day_calendar_constraints(slvr,stf,cal,sect):

    num_slots = len(WEEK_SLOTS)

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)

    for i in range(num_staff):
        sect_allstaff_idx = ALL_STAFF.index(staff[i])
        for j in range(num_slots):
            if cal[sect_allstaff_idx,j] > 0 or cal[sect_allstaff_idx,len(WEEK_SLOTS)+j/2] > 0:
                for k in range(num_shifts):
                    slvr.Add(stf[(k,j)] != i)

def set_call_calendar_constraints(slvr,stf,cal,sect):

    num_slots = len(WEEK_SLOTS)+len(CALL_SLOTS)

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)

    for i in range(num_staff):
        sect_allstaff_idx = ALL_STAFF.index(staff[i])
        for j in range(len(WEEK_SLOTS),CALL_SLOTS.index('SAT-AM')):
            if cal[sect_allstaff_idx,j] == ALL_SHIFTS.index('STAT3') or cal[sect_allstaff_idx,j] > 0:
                for k in range(num_shifts):
                    slvr.Add(stf[(k,j-len(WEEK_SLOTS))] != i)
        for j in range(CALL_SLOTS.index('SAT-AM'),num_slots):
            if cal[sect_allstaff_idx,j] > 0:
                for k in range(num_shifts):
                    slvr.Add(stf[(k,j-len(WEEK_SLOTS))] != i)

def set_brt_constraints(s,st): # s = solver

  for i in range(len(WEEK_SLOTS)):

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

    for i in range(len(WEEK_SLOTS)):

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
    s.Add(s.Max([st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),WEEK_SLOTS.index('TUE-AM'))] == SFL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)
    s.Add(s.Max([st[(SFL_SHIFTS.index('SLN_Sonoflu_AM'),WEEK_SLOTS.index('THU-AM'))] == SFL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)

def set_msk_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

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
    
    for i in range(len(WEEK_SLOTS)):

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
    
    for i in range(len(WEEK_SLOTS)):

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
    
    for i in range(len(WEEK_SLOTS)):

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
    
    for i in range(len(WEEK_SLOTS)):

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
    
    for i in range(len(WEEK_SLOTS)):

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
        s.Add(st[(OPR_SHIFTS.index('OPPR1_AM'),i*2+1)] == -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR2_AM'),i*2+1)] == -1)

def set_st3_constraints(s,st): # s = solver
    
    # STAT3 person is for the whole week
    for i in range(len(CALL_SLOTS)-5): # subtract the weekend days to get MON-THU (the last statement will be THU == FRI, that's why only 'til THU)
            s.Add(st[(ST3_SHIFTS.index('STAT3'),i)] == st[(ST3_SHIFTS.index('STAT3'),i+1)])
            
    for i in range(len(CALL_SLOTS)):

        if i < CALL_SLOTS.index('SAT-AM'): 
            # These shifts are real and need to be assigned (MON-FRI STAT3)
            s.Add(st[(ST3_SHIFTS.index('STAT3'),i)] != -1)
        else:
            # Shifts that don't fit into context (e.g. STAT3 not on weekends)
            s.Add(st[(ST3_SHIFTS.index('STAT3'),i)] == -1)

def set_swg_constraints(s,st): # s = solver
    
    # Only one Swing shift per week
    s.Add(s.AllDifferentExcept([st[(j,i)] for i in range(len(CALL_SLOTS)) for j in range(len(SWG_SHIFTS))],-1))

    for i in range(len(CALL_SLOTS)):

        if i < CALL_SLOTS.index('SAT-AM'): 
            # These shifts are real and need to be assigned (MON-FRI SWG)
            s.Add(st[(SWG_SHIFTS.index('Swing'),i)] != -1)
        else:
            # Shifts that don't fit into context (e.g. SWG not on weekends)
            s.Add(st[(SWG_SHIFTS.index('Swing'),i)] == -1)

def set_stw_constraints(s,st): # s = solver
    
    # Only one STAT shift per weekend
    s.Add(s.AllDifferentExcept([st[(j,i)] for i in range(len(CALL_SLOTS)) for j in range(len(STW_SHIFTS))],-1))
        
    for i in range(len(CALL_SLOTS)):
            if i < CALL_SLOTS.index('SAT-AM'):             
                # Shifts that don't fit into context (e.g. STATW on weekdays)
                s.Add(st[(STW_SHIFTS.index('STATW_AM'),i)] == -1)
                s.Add(st[(STW_SHIFTS.index('STATW_PM'),i)] == -1)
            elif i == CALL_SLOTS.index('SAT-AM') or i == CALL_SLOTS.index('SUN-AM'):
                s.Add(st[(STW_SHIFTS.index('STATW_AM'),i)] != -1)
                s.Add(st[(STW_SHIFTS.index('STATW_PM'),i)] == -1)
            else:
                s.Add(st[(STW_SHIFTS.index('STATW_AM'),i)] == -1)
                s.Add(st[(STW_SHIFTS.index('STATW_PM'),i)] != -1)

def set_wsp_constraints(s,st): # s = solver
    
    for i in range(len(CALL_SLOTS)):

        if i > CALL_SLOTS.index('FRI-PM'): 
            # Real shifts
            s.Add(st[(WSP_SHIFTS.index('WUSPR'),i)] != -1)
        else:
            # These shifts are not real (no WUSPR on weekdays)
            s.Add(st[(WSP_SHIFTS.index('WUSPR'),i)] == -1)

    # Constraints binding SAT/SUN rotations
    s.Add(st[(WSP_SHIFTS.index('WUSPR'),CALL_SLOTS.index('SAT-AM'))] == st[(WSP_SHIFTS.index('WUSPR'),CALL_SLOTS.index('SAT-PM'))])
    s.Add(st[(WSP_SHIFTS.index('WUSPR'),CALL_SLOTS.index('SAT-PM'))] == st[(WSP_SHIFTS.index('WUSPR'),CALL_SLOTS.index('SUN-AM'))])
    s.Add(st[(WSP_SHIFTS.index('WUSPR'),CALL_SLOTS.index('SUN-AM'))] == st[(WSP_SHIFTS.index('WUSPR'),CALL_SLOTS.index('SUN-PM'))])

def set_wmr_constraints(s,st): # s = solver
    
    for i in range(len(CALL_SLOTS)):
        if i > CALL_SLOTS.index('FRI-PM'): 
            # Real shifts
            s.Add(st[(WMR_SHIFTS.index('WMR'),i)] != -1)

        else:
            # These shifts are not real (no WMR on weekdays)
            s.Add(st[(WMR_SHIFTS.index('WMR'),i)] == -1)

    # Constraints binding SAT/SUN rotations
    s.Add(st[(WMR_SHIFTS.index('WMR'),CALL_SLOTS.index('SAT-AM'))] == st[(WMR_SHIFTS.index('WMR'),CALL_SLOTS.index('SAT-PM'))])
    s.Add(st[(WMR_SHIFTS.index('WMR'),CALL_SLOTS.index('SAT-PM'))] == st[(WMR_SHIFTS.index('WMR'),CALL_SLOTS.index('SUN-AM'))])
    s.Add(st[(WMR_SHIFTS.index('WMR'),CALL_SLOTS.index('SUN-AM'))] == st[(WMR_SHIFTS.index('WMR'),CALL_SLOTS.index('SUN-PM'))])

'''
====================
 ANALYSIS FUNCTIONS
====================
'''

def create_analysis(collect,stafflookup,cuml,hist,bias,sect):
    print("creating analysis...")

    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)

    analysis = []
    num_solutions = collect.SolutionCount()
    print("number of solutions:",num_solutions)
    for sol in range(num_solutions):
        curr = np.zeros((num_staff,num_shifts,num_slots))
        for i in range(num_slots):
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
            raise ValueError('Unresolved section in create_analysis function.')

        # sort by variance of each matrix; 
        analysis.append((sol,np.var(hist_plus),updated_cuml,hist_plus,curr))

    #print("sorting analysis of length", len(analysis))
    # finding the best choice of the array
    #analysis.sort(key=lambda x:x[1])
    return analysis[analysis.index(max(analysis,key=itemgetter(1)))]

    #return analysis

def create_call_analysis(collect,stafflookup,cuml,hist,bias,sect):
    print("creating analysis...")

    num_cshs = len(CALL_SLOTS)

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)

    analysis = []
    num_solutions = collect.SolutionCount()
    print("number of solutions:",num_solutions)
    for sol in range(num_solutions):
        curr = np.zeros((num_staff,num_shifts,num_cshs))
        for i in range(num_cshs):
            for j in range(num_shifts):
              st = collect.Value(sol,stafflookup[(j,i)])
              if st != -1: # if the rotation is covered by staff (not a placeholder halfday)
                  curr[st,j,i] += 1
        if sect == 'st3':
            updated_cuml,hist_plus = make_st3_hx(curr,cuml,hist,bias)
        elif sect == 'swg':
            updated_cuml,hist_plus = make_swg_hx(curr,cuml,hist,bias)
        elif sect == 'stw':
            updated_cuml,hist_plus = make_stw_hx(curr,cuml,hist,bias)
        elif sect == 'wsp':
            updated_cuml,hist_plus = make_wsp_hx(curr,cuml,hist,bias)
        elif sect == 'wmr':
            updated_cuml,hist_plus = make_wmr_hx(curr,cuml,hist,bias)
        else:
            raise ValueError('Unresolved section in create_call_analysis function.')

        # sort by variance of each matrix; 
        #analysis.append((sol,np.var(hist_plus),updated_cuml,hist_plus,curr))
        analysis.append((sol,np.sum(hist_plus),updated_cuml,hist_plus,curr))

    # finding the best fit
    #print("sorting analysis of length", len(analysis))
    print("finding the max of of this many potential solutions", len(analysis))
    #analysis.sort(key=lambda x:x[1],reverse=True)

    print("Analysis matrix max:", max(analysis,key=itemgetter(1))[1])
    print("Analysis matrix min:", min(analysis,key=itemgetter(1))[1])

    return analysis[analysis.index(max(analysis,key=itemgetter(1)))]

    #return analysis

def print_analysis(slvr,collect,stafflookup,anal,sect):
    print("printing analysis...")
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)
    
    #ts = anal[0][0] # ts = top solution
    ts = anal[0] # ts = top solution

    print()
    #print("Staffing matrix with variance:", anal[0][1])
    print("Staffing matrix with max value:", anal[1])
    for i in range(num_slots):
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
    #print("Variance max:", max(anal,key=itemgetter(1))[1])
    #print("Variance min:", min(anal,key=itemgetter(1))[1])
    print("Time:", slvr.WallTime(), "ms")
    print()

def print_call_analysis(slvr,collect,stafflookup,anal,sect):
    print("printing analysis...")
    num_cshs = len(CALL_SLOTS)

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)
    
    #ts = anal[0][0] # ts = top solution
    ts = anal[0]

    print()
    #print("Staffing matrix with variance:", anal[0][1])
    print("Staffing matrix with max value:", anal[1])
    for i in range(num_cshs):
        if i < CALL_SLOTS.index('SAT-AM'):
            print()
            print("Day", i)
            for j in range(num_shifts):
                st = collect.Value(ts,stafflookup[(j,i)])
                if st != -1:
                    print("WEEKDAY PM Call Shift:", shifts[j], staff[st])
        else:
            if i == CALL_SLOTS.index('SAT-AM'):
                for j in range(num_shifts):
                    st = collect.Value(ts,stafflookup[(j,i)])
                    if st != -1:
                        print(CALL_SLOTS[i],"AM Shift:", shifts[j], staff[st])
            elif i == CALL_SLOTS.index('SAT-PM'):
                for j in range(num_shifts):
                    st = collect.Value(ts,stafflookup[(j,i)])
                    if st != -1:
                        print(CALL_SLOTS[i],"PM Shift:", shifts[j], staff[st])
            elif i == CALL_SLOTS.index('SUN-AM'):
                for j in range(num_shifts):
                    st = collect.Value(ts,stafflookup[(j,i)])
                    if st != -1:
                        print(CALL_SLOTS[i],"AM Shift:", shifts[j], staff[st])
            elif i == CALL_SLOTS.index('SUN-PM'):
                for j in range(num_shifts):
                    st = collect.Value(ts,stafflookup[(j,i)])
                    if st != -1:
                        print(CALL_SLOTS[i],"PM Shift:", shifts[j], staff[st])
            else:
                pass
                        
    print()
    print("Solutions found:", collect.SolutionCount())
    #print("Variance max:", max(anal,key=itemgetter(1))[1])
    #print("Variance min:", min(anal,key=itemgetter(1))[1])
    print("Time:", slvr.WallTime(), "ms")
    print()

'''
=================
 PRINT FUNCTIONS
=================
'''

def print_calendar(cal):
    num_staff, num_slots, num_weeks = cal.shape

    for wk in range(num_weeks):
        print()
        print("===========================================")
        print("          WEEK #",wk)
        print("===========================================")
        print()
        line_header = '{:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18}'.format(CALLDAYS[0],CALLDAYS[1],CALLDAYS[2],CALLDAYS[3],CALLDAYS[4],CALLDAYS[5],CALLDAYS[6])
        print(line_header)
        for st in range(num_staff):
            print(ALL_STAFF[st])
            line_am = '{:>18} {:>18} {:>18} {:>18} {:>18}'.format(ALL_SHIFTS[cal[st,0,wk]],ALL_SHIFTS[cal[st,2,wk]],ALL_SHIFTS[cal[st,4,wk]],ALL_SHIFTS[cal[st,6,wk]],ALL_SHIFTS[cal[st,8,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM'),wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SUN-AM'),wk]])
            line_pm = '{:>18} {:>18} {:>18} {:>18} {:>18}'.format(ALL_SHIFTS[cal[st,1,wk]],ALL_SHIFTS[cal[st,3,wk]],ALL_SHIFTS[cal[st,5,wk]],ALL_SHIFTS[cal[st,7,wk]],ALL_SHIFTS[cal[st,9,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-PM'),wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SUN-PM'),wk]])
            line_call = '{:>18} {:>18} {:>18} {:>18} {:>18}'.format(ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+0,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+1,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+2,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+3,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+4,wk]])
            print(line_am)
            print(line_pm)
            print(line_call)

def print_results(results,section):
    num_staff,num_rots,staff,rots = get_section_nstaff_nrots_staff_rots(section) 

    if len(results.shape) > 2:
        print("Cumulative Week Summary for Section:", section)
        print("====================================")
        print()
        for s in range(num_staff):
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
        print()
        print("Current Bias Values for Section:", section, "(more negative numbers indicate rotation has not been covered by this staff member in a while)")
        print("================================")
        print()
        for s in range(num_staff):
            print("Staff",staff[s])
            for r in range(num_rots):
                alwk = results[s][r]
                print(rots[r],int(alwk))

def print_call_results(results,section):
    num_staff,num_rots,staff,rots = get_section_nstaff_nrots_staff_rots(section) 

    if len(results.shape) > 2:
        print("Cumulative Week Summary for Section:", section)
        print("====================================")
        print()
        for s in range(num_staff):
            print("Staff",staff[s])
            for r in range(num_rots):
                mon = results[s][r][CALLDAYS.index('MON')]
                tue = results[s][r][CALLDAYS.index('TUE')]
                wed = results[s][r][CALLDAYS.index('WED')]
                thu = results[s][r][CALLDAYS.index('THU')]
                fri = results[s][r][CALLDAYS.index('FRI')]
                sat = results[s][r][CALLDAYS.index('SAT')]
                sun = results[s][r][CALLDAYS.index('SUN')]
                alwk = mon+tue+wed+thu+fri+sat+sun
                print(rots[r],int(alwk)," MON",int(mon),"TUE",int(tue),"WED",int(wed),"THU",int(thu),"FRI",int(fri),"SAT",int(sat),"SUN",int(sun))
    else:
        print()
        print("Current Bias Values for Section:", section, "(more negative numbers indicate rotation has not been covered by this staff member in a while)")
        print("================================")
        print()
        for s in range(num_staff):
            print("Staff",staff[s])
            for r in range(num_rots):
                alwk = results[s][r]
                print(rots[r],int(alwk))

'''
=================
 BUILD FUNCTIONS
=================
'''

def build_brt(cal,cuml,hist,bias,limit):

    # Breast settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('brt')
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'brt')
    set_brt_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'brt')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'brt')

    return analysis[2],analysis[3],analysis[4]
    
def build_sfl(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('sfl')
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'sfl')
    set_sfl_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'sfl')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'sfl')

    return analysis[2],analysis[3],analysis[4]

def build_msk(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('msk')

    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'msk')
    set_msk_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'msk')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'msk')

    return analysis[2],analysis[3],analysis[4]

def build_ner(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('ner')
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'ner')
    set_ner_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'ner')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'ner')

    return analysis[2],analysis[3],analysis[4]

def build_abd(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('abd')
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'abd')
    set_abd_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'abd')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'abd')

    return analysis[2],analysis[3],analysis[4]

def build_cht(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('cht')
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'cht')
    set_cht_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'cht')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'cht')

    return analysis[2],analysis[3],analysis[4]

def build_sta(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('sta')
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'sta')
    set_sta_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'sta')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'sta')

    return analysis[2],analysis[3],analysis[4]

def build_opr(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('opr')
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'opr')
    set_opr_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'opr')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'opr')

    return analysis[2],analysis[3],analysis[4]

def build_st3(cal,cuml,hist,bias,limit):
    
    # ST3 settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('st3')
    num_cshs = len(CALL_SLOTS)
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_cshs,num_shifts,num_staff)

    # Constraints
    set_call_calendar_constraints(solver,staff,cal,'st3')
    set_st3_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_call_analysis(collector,staff,cuml,hist,bias,'st3')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'st3')

    return analysis[2],analysis[3],analysis[4]

def build_swg(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('swg')
    num_cshs = len(CALL_SLOTS)
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_cshs,num_shifts,num_staff)

    # Constraints
    set_call_calendar_constraints(solver,staff,cal,'swg')
    set_swg_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_call_analysis(collector,staff,cuml,hist,bias,'swg')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'swg')

    return analysis[2],analysis[3],analysis[4]

def build_stw(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('stw')
    num_slots = len(CALL_SLOTS)
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_call_calendar_constraints(solver,staff,cal,'stw')
    set_stw_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_call_analysis(collector,staff,cuml,hist,bias,'stw')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'stw')

    return analysis[2],analysis[3],analysis[4]

def build_wsp(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('wsp')
    num_slots = len(CALL_SLOTS)
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_call_calendar_constraints(solver,staff,cal,'wsp')
    set_wsp_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_call_analysis(collector,staff,cuml,hist,bias,'wsp')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'wsp')

    return analysis[2],analysis[3],analysis[4]

def build_wmr(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('wmr')
    num_slots = len(CALL_SLOTS)
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_call_calendar_constraints(solver,staff,cal,'wmr')
    set_wmr_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_call_analysis(collector,staff,cuml,hist,bias,'wmr')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'wmr')

    return analysis[2],analysis[3],analysis[4]

def build_other(cal,cuml,hist,bias):
    return cuml, hist

def build_multi_day(nweeks,sects,limit,calendar):
    
    ndays = len(WEEKDAYS)

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
            print()
            print("===========================================")
            print("          WEEK #",int(i+1)," ",sects[j])
            print("===========================================")
            
            if sects[j] == 'brt':      
                cumulative,history,recentweek = build_brt(calendar[:,:,i],cumulative,history,bias,limit) # recentweek is to update_calendar matrix
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'brt')
            elif sects[j] == 'sfl':
                cumulative,history,recentweek = build_sfl(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'sfl')
            elif sects[j] == 'msk':
                cumulative,history,recentweek = build_msk(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'msk')
            elif sects[j] == 'ner':
                cumulative,history,recentweek = build_ner(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'ner')
            elif sects[j] == 'abd':
                cumulative,history,recentweek = build_abd(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'abd')
            elif sects[j] == 'cht':
                cumulative,history,recentweek = build_cht(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'cht')
            elif sects[j] == 'sta':
                cumulative,history,recentweek = build_sta(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'sta')
            elif sects[j] == 'opr':
                cumulative,history,recentweek = build_opr(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'opr')
            else:
                currwk_cal = calendar[:,:,i]
                cumulative,history = build_other(currwk_cal,cumulative,history,bias)

            print_results(cumulative,sects[j])
            #print_results(history,sects[j])

    return calendar

def build_multi_call(nweeks,sects,limit,calendar):
    
    ncshs = len(CALL_SLOTS)

    for j in range(len(sects)):
        if sects[j] == 'st3':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('st3')            
            bias = init_st3_bias()
        elif sects[j] == 'swg':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('swg')      
            bias = init_swg_bias()
        elif sects[j] == 'stw':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('stw')      
            bias = init_stw_bias()
        elif sects[j] == 'wsp':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('wsp')      
            bias = init_wsp_bias()
        elif sects[j] == 'wmr':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('wmr')      
            bias = init_wmr_bias()
        else:
            raise ValueError('Unresolved section in build_multi_call function.')
            
        # cumulative and history are in the "rotation" context
        cumulative = np.zeros((nstaff,nrots,ncshs),dtype='int64') 
        history = np.zeros((nstaff,nrots),dtype='int64')

        for i in range(nweeks):
            print()
            print("===========================================")
            print("          WEEK #",int(i+1)," ",sects[j])
            print("===========================================")
            
            if sects[j] == 'st3':      
                cumulative,history,recentweek = build_st3(calendar[:,:,i],cumulative,history,bias,limit) # recentweek is to update_calendar matrix
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'st3')
            elif sects[j] == 'swg':
                cumulative,history,recentweek = build_swg(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'swg')
            elif sects[j] == 'stw':
                cumulative,history,recentweek = build_stw(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'stw')
            elif sects[j] == 'wsp':
                cumulative,history,recentweek = build_wsp(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'wsp')
            elif sects[j] == 'wmr':
                cumulative,history,recentweek = build_wmr(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'wmr')
            else:
                currwk_cal = calendar[:,:,i]
                cumulative,history = build_other(currwk_cal,cumulative,history,bias)

            print_call_results(cumulative,sects[j])
            #print_call_results(history,sects[j])

    return calendar

def update_calendar(cur,cal,sct): # c = nstaff x nhds reflecting 1-week (10 shift); a = calendar matrix
    
    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sct)

    for st in range(num_staff):
        for sh in range(num_shifts):
            for sl in range(len(WEEK_SLOTS)):
                if cur[st,sh,sl] > 0:
                    cal[ALL_STAFF.index(staff[st]),sl] = ALL_SHIFTS.index(shifts[sh])
    return cal

def update_call_calendar(cur,cal,sct): # c = nstaff x nhds reflecting 1-week (10 shift); a = calendar matrix

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sct)

    for st in range(num_staff):
        for sh in range(num_shifts):
            for sl in range(len(CALL_SLOTS)):
                if cur[st,sh,sl] > 0:
                    cal[ALL_STAFF.index(staff[st]),len(WEEK_SLOTS)+sl] = ALL_SHIFTS.index(shifts[sh])
    return cal 

        
def make_random_solver():
    # Creates the solver
    solver = pywrapcp.Solver("Schedule Solution")
    random.seed()
    r = int(random.random()*100000)
    print("random seed:", r)
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
    nhds = len(WEEK_SLOTS)
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
    nhds = len(WEEK_SLOTS)
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
    nhds = len(WEEK_SLOTS)
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
    nhds = len(WEEK_SLOTS)
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
    nhds = len(WEEK_SLOTS)
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
    nhds = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('cht')
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
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_sta_hx(cur,cml,his,bis):
    nhds = len(WEEK_SLOTS)
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
    nhds = len(WEEK_SLOTS)
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

def make_st3_hx(cur,cml,his,bis):
    ncshs = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('st3')

    curr_rots = np.zeros((nstaff,nrots,ncshs),dtype='int64')

    for s in range(nstaff):
        for i in range(ncshs):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    curr_rots[s,rots.index('STAT3'),i] += 1
                else:
                    pass
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_swg_hx(cur,cml,his,bis):
    ncshs = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('swg')

    curr_rots = np.zeros((nstaff,nrots,ncshs),dtype='int64')

    for s in range(nstaff):
        for i in range(ncshs):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    curr_rots[s,rots.index('Swing'),i] += 1
                else:
                    pass
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_stw_hx(cur,cml,his,bis):
    ncshs = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('stw')

    curr_rots = np.zeros((nstaff,nrots,ncshs),dtype='int64')

    for s in range(nstaff):
        for i in range(CALL_SLOTS.index('SAT-AM'),ncshs):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    # The rotation context works with days whereas the CALL_SLOTS split a morning evening (unlike the weekdays). 
                    # This is a bit-o-hack to convert the CALL_SHIFT array to the CALLDAYS array for the rotation context 
                    if i < CALL_SLOTS.index('SUN-AM'):
                        day_idx = CALLDAYS.index('SAT')
                    else:
                        day_idx = CALLDAYS.index('SUN')

                    if j == shifts.index('STATW_AM'):
                        curr_rots[s,rots.index('STATW_AM'),day_idx] += 1
                    elif j == shifts.index('STATW_PM'):
                        curr_rots[s,rots.index('STATW_PM'),day_idx] += 1
                    else:
                        pass
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_wsp_hx(cur,cml,his,bis):
    ncshs = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('wsp')

    curr_rots = np.zeros((nstaff,nrots,ncshs),dtype='int64')

    for s in range(nstaff):
        for i in range(ncshs):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if i > CALL_SLOTS.index('FRI-PM'):
                        curr_rots[s,rots.index('WUSPR'),i] += 1
                    else:
                        pass
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_wmr_hx(cur,cml,his,bis):
    ncshs = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('wmr')

    curr_rots = np.zeros((nstaff,nrots,ncshs),dtype='int64')

    for s in range(nstaff):
        for i in range(ncshs):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if i > CALL_SLOTS.index('FRI-PM'):
                        curr_rots[s,rots.index('WMR'),i] += 1
                    else:
                        pass
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis
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
    time_limit = 100
    day_sections = ['brt','sfl','ner','cht','msk','abd','sta','opr']
    #day_sections = ['brt']
    call_sections = []
    #call_sections = ['wmr']
    call_sections = ['st3','swg','stw','wsp','wmr']
    #call_sections = ['stw']
    #call_sections = ['swg']
    #sections = ['cht']
    #sections = ['sonoflu']

    # Used for keeping track of the current calendar for printing and for calendar
    calendar = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # calendar matrix is in the "shifts" context

    # Build multiphase call schedule
    if call_sections:
        calendar = build_multi_call(num_weeks,call_sections,time_limit,calendar)

    # Build multiphase weekday schedule
    if day_sections:
        calendar = build_multi_day(num_weeks,day_sections,time_limit,calendar)

    print_calendar(calendar)
        
if __name__ == "__main__":
  main()
