from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp
import numpy as np
from operator import itemgetter
import os,time,random
from schedsets import *
import qgendalysis as qa

'''
TODO
====
- improve multiphase print statements for rigorous testing
- experiment with sharing of special days, like UNCProcs (still not equally distributed); hierarchy of var?
- experiment with higher biases and adjust min value; what should the formula be? number of rotations/staff?
- revisit callback functions for decision tree to implement bias selection at that level
- proposed rules:
   1. spread out core rotations better, one person not more than 2-3 times per week
   2. pools only work so many days per week

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

def init_nuc_bias():
    return np.zeros((len(NUC_STAFF),len(NUC_ROTS)),dtype='int64') - 1

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

def init_scv_bias():
    return np.zeros((len(SCV_STAFF),len(SCV_ROTS)),dtype='int64') - 1

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

def set_staffweek(cal,initials,wk,reason):

    total_slots = len(WEEK_SLOTS)+len(CALL_SLOTS)

    for j in range(total_slots):
        cal[ALL_STAFF.index(initials),j,wk] = ALL_SHIFTS.index(reason)

def set_staffday(cal,initials,wk,day,reason):
    if day < len(WEEKDAYS): # block out a weekday
        cal[ALL_STAFF.index(initials),day*2,wk] = ALL_SHIFTS.index(reason) # AM shift
        cal[ALL_STAFF.index(initials),day*2+1,wk] = ALL_SHIFTS.index(reason) # PM shift
        cal[ALL_STAFF.index(initials),len(WEEK_SLOTS)+day,wk] = ALL_SHIFTS.index(reason) # PM call shift
    else: # block out the wknd
        cal[ALL_STAFF.index(initials),len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM'),wk] = ALL_SHIFTS.index(reason) # AM wknd call shift
        cal[ALL_STAFF.index(initials),len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM')+1,wk] = ALL_SHIFTS.index(reason) # PM wknd call shift

def set_staffshift(cal,initials,wk,day,slot,reason):
    if day < len(WEEKDAYS): # block out a weekday
        if slot == 'AM' or slot == 0:
            cal[ALL_STAFF.index(initials),day*2,wk] = ALL_SHIFTS.index(reason) # AM shift
        elif slot == 'PM' or slot == 1:
            cal[ALL_STAFF.index(initials),day*2+1,wk] = ALL_SHIFTS.index(reason) # PM shift
        elif slot == 'CALL' or slot == 2:
            cal[ALL_STAFF.index(initials),len(WEEK_SLOTS)+day,wk] = ALL_SHIFTS.index(reason) # PM call shift
    else:
        raise ValueError('Tried to block AM/PM weekend shift')

def set_day_calendar_constraints(slvr,stf,cal,sect):

    num_slots = len(WEEK_SLOTS)

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)

    for i in range(num_staff):
        sect_allstaff_idx = ALL_STAFF.index(staff[i])
        for j in range(num_slots):
            if cal[sect_allstaff_idx,j] > 0 or cal[sect_allstaff_idx,len(WEEK_SLOTS)+j/2] == ALL_SHIFTS.index('STAT3 4p-11p'): # the second term checks for STAT3 b/c can't work during the day if that's the case
                for k in range(num_shifts):
                    slvr.Add(stf[(k,j)] != i)

def set_call_calendar_constraints(slvr,stf,cal,sect):

    num_slots = len(WEEK_SLOTS)+len(CALL_SLOTS)
    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)
    
    print("Call calendar constraints for section: ",sect)
    for i in range(num_staff):
        blocked_wknd = False
        sect_allstaff_idx = ALL_STAFF.index(staff[i])

        # Handle STAT3 and Swing cases first (check if working day shifts during the week)
        if sect == 'st3':
            for j in range(len(WEEK_SLOTS)):
                if cal[sect_allstaff_idx,j] > 0:
                    for k in range(num_shifts):
                        slvr.Add(stf[(k,int(j/2))] != i) # index the PM shift rotations
        elif sect == 'swg':
            #print("handling Swing constraints")
            for j in range(len(WEEK_SLOTS)):
                if cal[sect_allstaff_idx,j] > 0 and cal[sect_allstaff_idx,j] < ALL_SHIFTS.index('Meeting'):
                    for k in range(num_shifts):
                        #print("leave Swing constraint:",k,int(j/2))                        
                        slvr.Add(stf[(k,int(j/2))] != i) # index the PM shift rotations
        else: # we are dealing with weekend rotation
            for j in range(len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM'),num_slots):
                if cal[sect_allstaff_idx,j] > 0:
                    blocked_wknd = True
            if blocked_wknd and sect in WKND_SECTS:
                for j in range(CALL_SLOTS.index('SAT-AM'),len(CALL_SLOTS)):
                    for k in range(num_shifts):
                        print("leave STATW constraint:",k,j,staff[i])                        
                        slvr.Add(stf[(k,j)] != i)

def set_brt_constraints(s,st): # s = solver

  for i in range(len(WEEK_SLOTS)):

      # No double coverage
      s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(BRT_SHIFTS))],-1))
      
  for i in range(len(WEEKDAYS)):

      # Constraints binding AM/PM rotations
      s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2)] == st[(BRT_SHIFTS.index('UCMam Diag 12-4p'),i*2+1)])
      s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2)] == st[(BRT_SHIFTS.index('UCMam Proc 12-4p'),i*2+1)])
      
      # Shifts that don't fit into context (e.g. UCMam Diag 12-4p on a morning shift)
      s.Add(st[(BRT_SHIFTS.index('UCMam Diag 12-4p'),i*2)] == -1)
      s.Add(st[(BRT_SHIFTS.index('UCMam Proc 12-4p'),i*2)] == -1)
      s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2+1)] == -1)
      s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2+1)] == -1)

      s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2)] != -1)
      s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2)] != -1)
      s.Add(st[(BRT_SHIFTS.index('UCMam Diag 12-4p'),i*2+1)] != -1)
      s.Add(st[(BRT_SHIFTS.index('UCMam Proc 12-4p'),i*2+1)] != -1)

      # Don't be on the same UNC rotation two days in a row (can relax if short-staffed)
      if i < 4:
          s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2)] != st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2+2)])
          s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2)] != st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2+2)])

  # Blocked Schedules (not all rotations are offered on every shift)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),0)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),1)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),2)] != -1)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),3)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),4)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),5)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),6)] != -1)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),7)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),8)] == -1)
  s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),9)] == -1)
  
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),0)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),1)] == -1)
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),2)] == -1)
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),3)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),4)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),5)] == -1)
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),6)] == -1)
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),7)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),8)] != -1)
  s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),9)] == -1)

def set_sfl_constraints(s,st): # s = solver
    
    # Don't cover the same Sonoflu shift in 1 week
    s.Add(s.AllDifferent([st[(j*2,i*2)] for j in range(len(SFL_SHIFTS)/2) for i in range(len(WEEKDAYS))]))

    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(SFL_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] == st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2+1)])
        s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] == st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != -1)
        s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] != -1)
        s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2+1)] != -1)
        s.Add(st[(SFL_SHIFTS.index('SLN_Sonoflu_PM'),i*2+1)] != -1)

        # Don't be on Sonoflu 2 days in a row
        if i < 4:
            # for same location
            s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2+2)])
            s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] != st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2+2)])
            
            # for different location
            s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] != st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2+2)])
            s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2+2)])

    # Only MSK person can cover SLN TUE/THU
    s.Add(s.Max([st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),WEEK_SLOTS.index('TUE-AM'))] == SFL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)
    s.Add(s.Max([st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),WEEK_SLOTS.index('THU-AM'))] == SFL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)

def set_msk_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(MSK_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(MSK_SHIFTS.index('MSK 8a-12p'),i*2)] == st[(MSK_SHIFTS.index('MSK 12-4p'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(MSK_SHIFTS.index('MSK 8a-12p'),i*2)] != -1)
        s.Add(st[(MSK_SHIFTS.index('MSK 12-4p'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(MSK_SHIFTS.index('MSK 12-4p'),i*2)] == -1)
        s.Add(st[(MSK_SHIFTS.index('MSK 8a-12p'),i*2+1)] == -1)

def set_abd_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(ABD_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(ABD_SHIFTS.index('Abdomen 8a-12p'),i*2)] == st[(ABD_SHIFTS.index('Abdomen 12-4p'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(ABD_SHIFTS.index('Abdomen 8a-12p'),i*2)] != -1)
        s.Add(st[(ABD_SHIFTS.index('Abdomen 12-4p'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(ABD_SHIFTS.index('Abdomen 12-4p'),i*2)] == -1)
        s.Add(st[(ABD_SHIFTS.index('Abdomen 8a-12p'),i*2+1)] == -1)

def set_ner_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(NER_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(NER_SHIFTS.index('Neuro 8a-12p'),i*2)] == st[(NER_SHIFTS.index('Neuro 12-4p'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(NER_SHIFTS.index('Neuro 8a-12p'),i*2)] != -1)
        s.Add(st[(NER_SHIFTS.index('Neuro 12-4p'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(NER_SHIFTS.index('Neuro 12-4p'),i*2)] == -1)
        s.Add(st[(NER_SHIFTS.index('Neuro 8a-12p'),i*2+1)] == -1)

def set_cht_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(CHT_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(CHT_SHIFTS.index('Chest/PET 8a-12p'),i*2)] == st[(CHT_SHIFTS.index('Chest/PET 12-4p'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(CHT_SHIFTS.index('Chest/PET 8a-12p'),i*2)] != -1)
        s.Add(st[(CHT_SHIFTS.index('Chest/PET 12-4p'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(CHT_SHIFTS.index('Chest/PET 12-4p'),i*2)] == -1)
        s.Add(st[(CHT_SHIFTS.index('Chest/PET 8a-12p'),i*2+1)] == -1)

def set_nuc_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(NUC_SHIFTS))],-1)) # ? whether this is necessary; should revisit
        
    for i in range(len(WEEKDAYS)):

        # Shifts that don't fit into context (e.g. Nucs not an AM shift)
        s.Add(st[(NUC_SHIFTS.index('Nucs 8a-4p'),i*2)] == -1)

        # The PM Nucs shift must be filled
        s.Add(st[(NUC_SHIFTS.index('Nucs 8a-4p'),i*2+1)] != -1)

def set_sta_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(STA_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2)] == st[(STA_SHIFTS.index('STAT1b 12p-4p'),i*2+1)])

        # These shifts are real and need to be assigned
        s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2)] != -1)
        s.Add(st[(STA_SHIFTS.index('STAT1b 12p-4p'),i*2+1)] != -1)
        s.Add(st[(STA_SHIFTS.index('STAT2 12p-4p'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(STA_SHIFTS.index('STAT1b 12p-4p'),i*2)] == -1)
        s.Add(st[(STA_SHIFTS.index('STAT2 12p-4p'),i*2)] == -1)
        s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2+1)] == -1)

        # Don't be on all day STAT two days in a row 
        if i < 4:
            s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2)] != st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2+2)])

def set_opr_constraints(s,st): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(OPR_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # These shifts are real and need to be assigned
        s.Add(st[(OPR_SHIFTS.index('OPPR1am'),i*2)] != -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR2am'),i*2)] != -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR3pm'),i*2+1)] != -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR4pm'),i*2+1)] != -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(OPR_SHIFTS.index('OPPR3pm'),i*2)] == -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR4pm'),i*2)] == -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR1am'),i*2+1)] == -1)
        s.Add(st[(OPR_SHIFTS.index('OPPR2am'),i*2+1)] == -1)

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
                s.Add(st[(STW_SHIFTS.index('STATWAM 8a-330p'),i)] == -1)
                s.Add(st[(STW_SHIFTS.index('STATWPM 330p-11p'),i)] == -1)
            elif i == CALL_SLOTS.index('SAT-AM') or i == CALL_SLOTS.index('SUN-AM'):
                s.Add(st[(STW_SHIFTS.index('STATWAM 8a-330p'),i)] != -1)
                s.Add(st[(STW_SHIFTS.index('STATWPM 330p-11p'),i)] == -1)
            else:
                s.Add(st[(STW_SHIFTS.index('STATWAM 8a-330p'),i)] == -1)
                s.Add(st[(STW_SHIFTS.index('STATWPM 330p-11p'),i)] != -1)

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

def set_scv_constraints(s,st): # s = solver

    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(SCV_SHIFTS))],-1))

    # On Mondays set having an NEU, MSK, and ABD/CHT SCV 
    s.Add(s.Max([st[(SCV_SHIFTS.index('SCV1_AM'),WEEK_SLOTS.index('MON-AM'))] == SCV_STAFF.index(rad) for rad in NER_STAFF]) == 1)
    s.Add(s.Max([st[(SCV_SHIFTS.index('SCV2_AM'),WEEK_SLOTS.index('MON-AM'))] == SCV_STAFF.index(rad) for rad in MSK_STAFF]) == 1)
    s.Add(s.Max([st[(SCV_SHIFTS.index('SCV3_AM'),WEEK_SLOTS.index('MON-AM'))] == SCV_STAFF.index(rad) for rad in ABD_STAFF]) == 1)
    
    for i in range(len(WEEKDAYS)):
        
        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(SCV_SHIFTS.index('SCV1_PM'),i*2)] == -1)
        s.Add(st[(SCV_SHIFTS.index('SCV2_PM'),i*2)] == -1)
        s.Add(st[(SCV_SHIFTS.index('SCV1_AM'),i*2+1)] == -1)
        s.Add(st[(SCV_SHIFTS.index('SCV2_AM'),i*2+1)] == -1)
        s.Add(st[(SCV_SHIFTS.index('SCV3_AM'),i*2+1)] == -1)
                
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
        elif sect == 'nuc':
            updated_cuml,hist_plus = make_nuc_hx(curr,cuml,hist,bias)
        elif sect == 'sta':
            updated_cuml,hist_plus = make_sta_hx(curr,cuml,hist,bias)
        elif sect == 'opr':
            updated_cuml,hist_plus = make_opr_hx(curr,cuml,hist,bias)
        elif sect == 'scv':
            updated_cuml,hist_plus = make_scv_hx(curr,cuml,hist,bias)
        else:
            raise ValueError('Unresolved section in create_analysis function.')

        # sort matrix by certain criteria
        #analysis.append((sol,np.var(hist_plus),updated_cuml,hist_plus,curr))
        analysis.append((sol,np.sum(hist_plus),updated_cuml,hist_plus,curr))

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

        # sort matrix by certain criteria
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
===================
 RESULTS FUNCTIONS
===================
'''

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
====================
 CALENDAR FUNCTIONS
====================
'''

def convert_staff_to_shift_calendar(stfcal):

    num_slots = len(WEEK_SLOTS)+len(CALL_SLOTS)
    num_weeks = stfcal.shape[2]
    num_staff = stfcal.shape[0]

    shftcal = np.zeros((len(ALL_SHIFTS),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64')
    shftcal = shftcal - 1 # -1 means no staff covering

    for week in range(num_weeks):
        for shift in range(len(ALL_SHIFTS)):
            for slot in range(num_slots):
                for stf in range(num_staff):
                    if stfcal[stf,slot,week] == shift:
                        shftcal[shift,slot,week] = stf

    return shftcal

def print_shift_calendar(cal):
    num_shifts, num_slots, num_weeks = cal.shape

    for wk in range(num_weeks):
        print()
        print("===========================================")
        print("          WEEK #",wk)
        print("===========================================")
        print()
        line_header = '{:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18}'.format(CALLDAYS[0],CALLDAYS[1],CALLDAYS[2],CALLDAYS[3],CALLDAYS[4],CALLDAYS[5],CALLDAYS[6])
        print(line_header)
        for sh in range(1,num_shifts): # skip 0 b/c it is the "no assigment" indicator
            print(ALL_SHIFTS[sh])
            line_am = ""
            line_pm = ""
            line_call = ""
            for sl in range(num_slots):
                if cal[sh,sl,wk] < 0:
                    staff = '----'
                else:
                    staff = ALL_STAFF[cal[sh,sl,wk]]
                if sl < len(WEEK_SLOTS):
                    if sl%2 == 0:
                        line_am += '{:>18}'.format(staff)
                    else:
                        line_pm += '{:>18}'.format(staff)
                elif sl < len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM'):
                    line_call += '{:>18}'.format(staff)
                else:
                    if sl%2 == 1: # AM wknd slot
                        line_am += '{:>18}'.format(staff)
                    else: # PM wknd slot
                        line_pm += '{:>18}'.format(staff)
            print(line_am)
            print(line_pm)
            print(line_call)

def print_staff_calendar(cal):
    num_staff, num_slots, num_weeks = cal.shape

    #for wk in range(num_weeks):
    for wk in range(2): # for testing just print the first few weeks
        print()
        print("===========================================")
        print("          WEEK #",wk)
        print("===========================================")
        print()
        line_header = '{:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}'.format(CALLDAYS[0],CALLDAYS[1],CALLDAYS[2],CALLDAYS[3],CALLDAYS[4],CALLDAYS[5],CALLDAYS[6])
        print(line_header)
        for st in range(num_staff):
            print(ALL_STAFF[st])
            line_am = '{:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}'.format(ALL_SHIFTS[cal[st,0,wk]],ALL_SHIFTS[cal[st,2,wk]],ALL_SHIFTS[cal[st,4,wk]],ALL_SHIFTS[cal[st,6,wk]],ALL_SHIFTS[cal[st,8,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM'),wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SUN-AM'),wk]])
            line_pm = '{:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}'.format(ALL_SHIFTS[cal[st,1,wk]],ALL_SHIFTS[cal[st,3,wk]],ALL_SHIFTS[cal[st,5,wk]],ALL_SHIFTS[cal[st,7,wk]],ALL_SHIFTS[cal[st,9,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-PM'),wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SUN-PM'),wk]])
            line_call = '{:>25} {:>25} {:>25} {:>25} {:>25}'.format(ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+0,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+1,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+2,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+3,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+4,wk]])
            print(line_am)
            print(line_pm)
            print(line_call)

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

def build_nuc(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('nuc')
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'nuc')
    set_nuc_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'nuc')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'nuc')

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

def build_scv(cal,cuml,hist,bias,limit):
    
    # Sonoflu settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts('scv')
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create staff lookup
    staff, staff_flat = create_staff_lookup(solver,num_slots,num_shifts,num_staff)

    # Constraints
    set_day_calendar_constraints(solver,staff,cal,'scv')
    set_scv_constraints(solver,staff)

    # Creating decision builder and collector
    collector = get_collector(solver,staff_flat,time_limit)

    # analyze and sort results based on schedule variance
    analysis = create_analysis(collector,staff,cuml,hist,bias,'scv')

    # Print out the top solution with the least variance
    print_analysis(solver,collector,staff,analysis,'scv')

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
        elif sects[j] == 'nuc':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('nuc')      
            bias = init_nuc_bias()
        elif sects[j] == 'sta':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('sta')      
            bias = init_sta_bias()
        elif sects[j] == 'opr':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('opr')      
            bias = init_opr_bias()
        elif sects[j] == 'scv':
            nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots('scv')      
            bias = init_scv_bias()
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
            elif sects[j] == 'nuc':
                cumulative,history,recentweek = build_nuc(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'nuc')
            elif sects[j] == 'sta':
                cumulative,history,recentweek = build_sta(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'sta')
            elif sects[j] == 'opr':
                cumulative,history,recentweek = build_opr(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'opr')
            elif sects[j] == 'scv':
                cumulative,history,recentweek = build_scv(calendar[:,:,i],cumulative,history,bias,limit)
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],'scv')
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
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)

    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('brt')
    
    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j < 2 and i%2 == 1: # the UNC-Diag AM/PM are both considered UNC-Diag
                        curr_rots[s,rots.index('UNC_Diag'),int(i/2)] += 1
                    elif j < 4 and i%2 == 1:  # the UNC-Proc AM/PM are both considered UNC-Proc
                        curr_rots[s,rots.index('UNC_Proc'),int(i/2)] += 1
                    elif j == 4: # FRE_Mamm
                        curr_rots[s,rots.index('FRE_Mamm'),int(i/2)] += 1
                    elif j == 5: # SLN Mamm
                        curr_rots[s,rots.index('SLN_Mamm'),int(i/2)] += 1
                    else:
#raise ValueError('Unresolved shift/halfday combination in make_brt_hx function.',i,j)
                        pass
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_sfl_hx(cur,cml,his,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('sfl')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Fre US/Fluoro 8a-4p') and i%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('FRE_Sonoflu'),int(i/2)] += 1
                    elif j == shifts.index('SL US/Fluoro 8a-4p') and i%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('SLN_Sonoflu'),int(i/2)] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_sfl_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_msk_hx(cur,cml,his,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('msk')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
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
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('abd')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
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
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('ner')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
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
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('cht')
    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
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

def make_nuc_hx(cur,cml,his,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('nuc')
    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Nucs') and i%2 == 1: # nucs is a PM rotation only
                        curr_rots[s,rots.index('Nucs'),int(i/2)] += 1
                    else:
                        pass

    new_cml = cml.astype('int64')+curr_rots.astype('int64')
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_sta_hx(cur,cml,his,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('sta')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
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
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('opr')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
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

def make_scv_hx(cur,cml,his,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('scv')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0: # any SCV rotation whether AM/PM counts as one rotation
                    curr_rots[s,rots.index('SCV'),int(i/2)] += 1
                else:
                    pass
                
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_st3_hx(cur,cml,his,bis):
    nslts = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('st3')

    curr_rots = np.zeros((nstaff,nrots,nslts),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    curr_rots[s,rots.index('STAT3'),i] += 1
                else:
                    pass
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_swg_hx(cur,cml,his,bis):
    nslts = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('swg')

    curr_rots = np.zeros((nstaff,nrots,nslts),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    curr_rots[s,rots.index('Swing'),i] += 1
                else:
                    pass
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    hist_plus = add_history_matrix(his,np.sum(curr_rots,axis=2).astype('int64'))+bis

    return new_cml,hist_plus

def make_stw_hx(cur,cml,his,bis):
    nslts = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('stw')

    curr_rots = np.zeros((nstaff,nrots,nslts),dtype='int64')

    for s in range(nstaff):
        for i in range(CALL_SLOTS.index('SAT-AM'),nslts):
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
    nslts = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('wsp')

    curr_rots = np.zeros((nstaff,nrots,nslts),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
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
    nslts = len(CALL_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('wmr')

    curr_rots = np.zeros((nstaff,nrots,nslts),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
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
    day_sections = ['nuc','brt','sfl','ner','cht','msk','abd','sta','scv','opr']
    #day_sections = []
    #call_sections = []
    #call_sections = ['swg','stw']
    call_sections = ['st3','swg','stw','wsp','wmr']
    #call_sections = ['stw']
    #call_sections = ['swg']
    #sections = ['cht']
    #sections = ['sonoflu']
    fname = '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/Staff_2017.csv'

    # Get the department information from file
    dept = qa.load_data(fname)
    staff_calendar = qa.qgimport(dept).astype('int64')

    # Used for keeping track of the schedule by staff
    #staff_calendar = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # staff_calendar matrix is in the "slots" context

    # Set staff_calendar constraints
    '''set_staffweek(staff_calendar,'CCM',0,'Leave')
    set_staffweek(staff_calendar,'SMN',0,'Leave')
    set_staffday(staff_calendar,'JDB',0,0,'Admin')
    set_staffday(staff_calendar,'SDE',0,2,'Admin')
    set_staffshift(staff_calendar,'GJS',0,3,0,'Admin')'''

    # Build multiphase call schedule
    '''if call_sections:
        staff_calendar = build_multi_call(num_weeks,call_sections,time_limit,staff_calendar)

    # Build multiphase weekday schedule
    if day_sections:
        staff_calendar = build_multi_day(num_weeks,day_sections,time_limit,staff_calendar)'''

    print_staff_calendar(staff_calendar)
    #shift_calendar = convert_staff_to_shift_calendar(staff_calendar)
    #print_shift_calendar(shift_calendar)        

if __name__ == "__main__":
  main()
