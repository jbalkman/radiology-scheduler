from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp
import numpy as np
from operator import itemgetter
import os,time,random
from schedsets import *
import qgendalysis as qa
import csv

'''
TODO
====
- vacation preference scheduling
- weighted factor to include X weeks of history (in addition to counter cost minimization)
- force certain number of rotations for certain staff (HG Sonoflu); handle using the bias?
- fix "rotation" counting in the set_rotation_constraints function to be by the day for day shifts instead of by the AM/PM
- consolidate rotations OPPR, Sonoflu (don't need to split these into separate rotations, except OPPR is a halfday and Sonoflu is a whole day)
- prevent sequenstering of favored rotations
- use cumulative matrix to have some sort of moving avg that equalizes rotations (input to bias?)

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
CHOOSE_MIN_SIZEXF
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

'''
=================
 GENERAL PURPOSE
==================
'''

# function currently not used but serves as an example for adding constraints for staff leave/vacation
# r = staff; a = leave day(s) list; s = solver; l = leave_days variable, st = staff variable; ns = num_shifts
def leave(r,a,s,st,ns):
	rad = BRT_STAFF.index(r)
	for d in a:
		s.Add(s.Max([st[(k,d*2)] == rad for k in range(ns)]) == 0)
		s.Add(s.Max([st[(k,d*2+1)] == rad for k in range(ns)]) == 0)

def juggle_sections(original,failed):
    fail_idx = original.index(failed)

    if fail_idx > 0:
        if failed == 'scv':
            # just move the failed section up by one
            original[fail_idx-1], original[fail_idx] = original[fail_idx], original[fail_idx-1]
        else:
            # move the failed section up to the front
            original.insert(0,original.pop(fail_idx))

        return original
    else:
        sys.exit("Failed on the first section indicating strict constraint problem or wasn't given enough time to find solution (lengthen time limit).")
        
'''
=====================
 CP SOLVER VARIABLES
=====================
'''

def create_staff_lookup(solver,num_hdays,num_shifts,num_staff):
    staff = {}
    
    # the staff matrix returns staff for a given slot and given shift
    for i in range(num_hdays):
        for j in range(num_shifts):
            staff[(j,i)] = solver.IntVar(-1, num_staff - 1, "staff(%i,%i)" % (j, i)) # -1 is an escape where shift to not applicable to time of day
    staff_flat = [staff[(j, i)] for j in range(num_shifts) for i in range(num_hdays)]

    return staff, staff_flat

def create_staff_counts_lookup(solver,num_hdays,section):
    staff = {}
    
    #num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(section)
    num_staff,num_shifts,_,staff_tup,shifts_tup,_ = get_section_info(section)

    # the staff matrix returns staff for a given slot and given shift
    for i in range(num_hdays):
        for j in range(num_shifts):
            staff[(j,i)] = solver.IntVar(-1, num_staff - 1, "staff(%i,%i)" % (j, i)) # -1 is an escape where shift to not applicable to time of day

    staff_flat = [staff[(j, i)] for j in range(num_shifts) for i in range(num_hdays)]

    max_val = 100 # just set very high
    #scounts = [solver.IntVar(0, max_val, "scount[%i]" % s) for s in range(num_staff)] # overall staff counts (shift slots during the period)
    #pcounts = [solver.IntVar(0, max_val, "pcount[%i]" % s) for s in range(num_staff)] # overall staff counts (shift slots during the period)
    #pcounts = solver.IntVar(0, max_val, "pcounts")
    
    '''for s in range(num_staff):
        print("curr staff:",s)
        #scounts[s] = solver.Sum([solver.IsEqualCstVar(staff[(i,j)],ALL_STAFF.index(staff_tup[s])) for i in range(num_hdays) for j in range(num_shifts)])
        scounts[s] = solver.Sum([solver.IsEqualCstVar(staff[(i,j)],ALL_STAFF.index(staff_tup[s])) for i in range(1) for j in range(1)])'''
    
    poolidx = [staff_tup.index(i) for i in staff_tup if i in POOLS]
    #print("poolidx",poolidx)
    pcounts = solver.Sum([solver.IsMemberVar(staff[(j,i)],poolidx) for i in range(num_hdays) for j in range(num_shifts)])
    
    #staffidx = [solver.IntVar(0, len(ALL_STAFF), "staffidx(%i)" % (i)) for i in range(num_staff)]
    #pcounts = solver.Sum([solver.ScalProd(scounts[s],solver.IsMemberVar(ALL_STAFF.index(staff_tup[s]),POOLS)) for s in range(num_staff)]) # overall pool counts

    return staff, staff_flat, pcounts

def create_variables(solver,nslots,section):
    v_staff = {}
    v_rots = {}
    v_cntr = {}
    v_rotprod = {}

    v_tcost = solver.IntVar(-500,500, "v_tcost")

    #num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(section)
    #_,_,num_rots,_,rots_tup = get_section_nstaff_nshifts_nrots_shifts_rots(section)

    num_staff,num_shifts,num_rots,staff_tup,shifts_tup,rots_tup = get_section_info(section)

    # the staff matrix returns staff for a given slot and given shift
    for i in range(nslots):
        for j in range(num_shifts):
            v_staff[(j,i)] = solver.IntVar(-1, num_staff-1, "staff(%i,%i)" % (j, i)) # -1 is an escape where shift to not applicable to time of day

    # represents the matrix to be optimized
    for j in range(num_staff):
        for i in range(num_rots):
            v_rots[(j,i)] = solver.IntVar(-1000,1000, "rots(%i,%i)" % (j, i))
            v_cntr[(j,i)] = solver.IntVar(-1000,1000, "v_cntr(%i,%i)" % (j, i))
            v_rotprod[(j,i)] = solver.IntVar(0,1, "v_rotprod(%i,%i)" % (j, i))

    # flattened versions 
    v_staff_flat = [v_staff[(j, i)] for j in range(num_shifts) for i in range(nslots)]
    v_rots_flat = [v_rots[(j, i)] for j in range(num_staff) for i in range(num_rots)]
    v_cntr_flat = [v_cntr[(stf,rot)] for stf in range(num_staff) for rot in range(num_rots)]
    v_rotprod_flat = [v_rotprod[(stf,rot)] for stf in range(num_staff) for rot in range(num_rots)]

    max_val = 100 # just set very high
    #scounts = [solver.IntVar(0, max_val, "scount[%i]" % s) for s in range(num_staff)] # overall staff counts (shift slots during the period)
    #pcounts = [solver.IntVar(0, max_val, "pcount[%i]" % s) for s in range(num_staff)] # overall staff counts (shift slots during the period)
    #pcounts = solver.IntVar(0, max_val, "pcounts")
    
    '''for s in range(num_staff):
        print("curr staff:",s)
        #scounts[s] = solver.Sum([solver.IsEqualCstVar(staff[(i,j)],ALL_STAFF.index(staff_tup[s])) for i in range(nslots) for j in range(num_shifts)])
        scounts[s] = solver.Sum([solver.IsEqualCstVar(staff[(i,j)],ALL_STAFF.index(staff_tup[s])) for i in range(1) for j in range(1)])'''
    
    poolidx = [staff_tup.index(i) for i in staff_tup if i in LCM_STAFF]
    v_pcounts = solver.Sum([solver.IsMemberVar(v_staff[(j,i)],poolidx) for i in range(nslots) for j in range(num_shifts)])

    return v_staff,v_staff_flat,v_rots,v_rots_flat,v_cntr,v_cntr_flat,v_rotprod,v_rotprod_flat,v_tcost,v_pcounts

def create_pool_shifts(solver,nweeks,pools):
    shifts = {}
    
    for p in range(len(pools)):
        nshifts = len(ALL_SHIFTS)
        #print("Number of possible shifts:",nshifts)
        for w in range(nweeks):
            for s in range(len(WEEK_SLOTS)):
                #print("Creating variables for j,k,l tuple:",pools[p][0],p,w,s)
                shifts[(p,w,s)] = solver.IntVar(-1, nshifts, "shifts(%i,%i,%i)" % (p,w,s)) # -1 is an escape; 1 is for PM shift
    shifts_flat = [shifts[(p,w,s)] for p in range(len(pools)) for w in range(nweeks) for s in range(len(WEEK_SLOTS))]

    return shifts, shifts_flat

def create_pooltba_shifts(solver):
    shifts = {}

    v_tcost = solver.IntVar(-500,500, "v_tcost") # here the cost variable tries to minimize the TBA shifts
    
    for p in range(len(LCM_STAFF)):
        nshifts = len(ALL_SHIFTS)

        for s in range(len(WEEK_SLOTS)):
		# unlike other shift variables, the escape can be "0" here instead of "-1" b/c we are assigning shifts wrt to ALL_SHIFTS, where [0] = ""
		shifts[(p,s)] = solver.IntVar(0, nshifts-1, "shifts(%i,%i)" % (p,s))
    shifts_flat = [shifts[(p,s)] for p in range(len(LCM_STAFF)) for s in range(len(WEEK_SLOTS))]

    return shifts, shifts_flat, v_tcost

def create_neshifts(solver,nslots,section):
    v_neshifts = {}
    v_rots = {}
    v_cntr = {}
    v_rotprod = {}

    v_tcost = solver.IntVar(-500,500, "v_tcost")

    num_staff,num_neshifts,num_rots,staff_tup,neshifts_tup,rots_tup = get_section_info(section)
    #num_staff,num_neshifts,staff_tup,neshifts_tup = get_section_nstaff_nshifts_staff_shifts(section)
    #_,_,num_rots,_,rots_tup = get_section_nstaff_nshifts_nrots_shifts_rots(section)

    for stf in range(num_staff):
        for slt in range(len(WEEK_SLOTS)):
            v_neshifts[(stf,slt)] = solver.IntVar(-1, num_neshifts-1, "neshifts(%i,%i)" % (stf,slt))
    v_neshifts_flat = [v_neshifts[(i,j)] for i in range(num_staff) for j in range(len(WEEK_SLOTS))]

    # represents the matrix to be optimized
    for j in range(num_staff):
        for i in range(num_rots):
            v_rots[(j,i)] = solver.IntVar(-1000,1000, "rots(%i,%i)" % (j, i))
            v_cntr[(j,i)] = solver.IntVar(-1000,1000, "v_cntr(%i,%i)" % (j, i))
            v_rotprod[(j,i)] = solver.IntVar(0,1, "v_rotprod(%i,%i)" % (j, i))

    # flattened versions 
    v_rots_flat = [v_rots[(j,i)] for j in range(num_staff) for i in range(num_rots)]
    v_cntr_flat = [v_cntr[(stf,rot)] for stf in range(num_staff) for rot in range(num_rots)]
    v_rotprod_flat = [v_rotprod[(stf,rot)] for stf in range(num_staff) for rot in range(num_rots)]

    return v_neshifts,v_neshifts_flat,v_rots,v_rots_flat,v_cntr,v_cntr_flat,v_rotprod,v_rotprod_flat,v_tcost

'''
================
 BIAS FUNCTIONS
================
'''

def init_rcounters(cumulative,counter,section):
    #nstaff,nrots,staff_tup,rots_tup = get_section_nstaff_nrots_staff_rots(section)  
    nstaff,_,nrots,staff_tup,_,rots_tup = get_section_info(section)  

    r_cumulative = np.zeros((nstaff,nrots),dtype='int64')
    r_counter = np.zeros((nstaff,nrots),dtype='int64')
    r_ocounter = np.zeros((nstaff,nrots),dtype='int64')

    for s in range(nstaff):
        for r in range(nrots):
            r_counter[s,r] = counter[ALL_STAFF.index(staff_tup[s]),ALL_ROTS.index(rots_tup[r])]
            r_cumulative[s,r] = cumulative[ALL_STAFF.index(staff_tup[s]),ALL_ROTS.index(rots_tup[r])]

    return r_cumulative, (r_counter, r_ocounter)

def init_counter_history(cal,cuml,cntr):
    print("initializing counter history...")

    bias = init_general_bias()
    nweeks = cal.shape[2]

    for wk in range(nweeks):
        cuml,cntr = make_week_hx(cal[:,:,wk],cuml,cntr,bias)
    return cuml,cntr,bias

def set_bias_offset(bias,cumulative,section):
    #nstaff,nrots,staff,rots = get_section_nstaff_nrots_staff_rots(section)            
    nstaff,_,nrots,staff,_,rots = get_section_info(section)            
    offset_value = 1
    threshold = 2

    for r in range(nrots):
        sidx = np.array([ALL_STAFF.index(staff[s]) for s in range(nstaff) if (ALL_STAFF.index(staff[s]) not in LCM_STAFF)])
        ridx = np.repeat(ALL_ROTS.index(rots[r]),len(sidx))
        mean = np.mean(cumulative[sidx,ridx])

        for s in range(nstaff):
            if ALL_STAFF.index(staff[s]) not in LCM_STAFF: # don't adjust bias for pools
                curr = cumulative[ALL_STAFF.index(staff[s]),ALL_ROTS.index(rots[r])] 
                if curr > mean + threshold:
                    bias[s,r] -= offset_value
                elif curr < mean - threshold:
                    bias[s,r] += offset_value

def get_bias(section,cumulative=None):

    if section == 'brt':
        bias = init_brt_bias()
    elif section == 'sfl':
        bias = init_sfl_bias()
    elif section == 'msk':
        bias = init_msk_bias()
    elif section == 'ner':
        bias = init_ner_bias()
    elif section == 'abd':
        bias = init_abd_bias()
    elif section == 'cht':
        bias = init_cht_bias()
    elif section == 'nuc':
        bias = init_nuc_bias()
    elif section == 'sta':
        bias = init_sta_bias()
    elif section == 'opr':
        bias = init_opr_bias()
    elif section == 'scv':
        bias = init_scv_bias()
    elif section == 'adm':
        bias = init_adm_bias()
    elif section == 'st3':
	bias = init_st3_bias()
    elif section == 'swg':
	bias = init_swg_bias()
    elif section == 'stw':
	bias = init_stw_bias()
    elif section == 'wsp':
	bias = init_wsp_bias()
    elif section == 'wmr':
	bias = init_wmr_bias()
    elif section == 'nhk':
	bias = init_nhk_bias()
    else:
        pass

    # Use the cumulative matrix to offset the bias
    if cumulative is None:
        pass
    else:
        set_bias_offset(bias,cumulative,section)
        
    return bias

def init_general_bias():
    bias = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64')
    
    sections = ['brt','cht','nuc','sfl','msk','abd','ner','sta','scv','opr','adm']    
    
    for sect in sections:
        #nstaff, nrots, staff_tup, rots_tup = get_section_nstaff_nrots_staff_rots(sect)
        nstaff,_,nrots,staff_tup,_,rots_tup = get_section_info(sect)
        staff_idx = [ALL_STAFF.index(staff_tup[s]) for s in range(len(staff_tup))]
        rots_idx = [ALL_ROTS.index(rots_tup[r]) for r in range(len(rots_tup))]
        for si in staff_idx:
            if ALL_STAFF[si] not in LCM_STAFF:
                for ri in rots_idx:
                    bias[si,ri] += 1

    return bias

def init_brt_bias():
    bias = np.zeros((len(BRT_STAFF),len(BRT_ROTS)),dtype='int64') + 1 # here the bias is -2 for all rotations; may need to be less for rotations that are less frequent (e.g. -1 for SLN_Mamm)

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in BRT_STAFF:
            bias[BRT_STAFF.index(LCM_STAFF[i]),:] = bias[BRT_STAFF.index(LCM_STAFF[i]),:] - 3
    return bias

def init_sfl_bias():
    bias = np.zeros((len(SFL_STAFF),len(SFL_ROTS)),dtype='int64') + 1

    # bias HG towards fluoro
    bias[SFL_STAFF.index('HG'),:] = bias[SFL_STAFF.index('HG'),:] + 3

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in SFL_STAFF:
            bias[SFL_STAFF.index(LCM_STAFF[i]),:] = bias[SFL_STAFF.index(LCM_STAFF[i]),:] - 3
    return bias

def init_msk_bias():
    bias = np.zeros((len(MSK_STAFF),len(MSK_ROTS)),dtype='int64') + 1

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in MSK_STAFF:
            bias[MSK_STAFF.index(LCM_STAFF[i]),:] = bias[MSK_STAFF.index(LCM_STAFF[i]),:] - 3
    return bias

def init_ner_bias():
    bias = np.zeros((len(NER_STAFF),len(NER_ROTS)),dtype='int64') + 1

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in NER_STAFF:
            bias[NER_STAFF.index(LCM_STAFF[i]),:] = bias[NER_STAFF.index(LCM_STAFF[i]),:] - 3
    return bias

def init_abd_bias():
    bias = np.zeros((len(ABD_STAFF),len(ABD_ROTS)),dtype='int64') + 1

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in ABD_STAFF:
            bias[ABD_STAFF.index(LCM_STAFF[i]),:] = bias[ABD_STAFF.index(LCM_STAFF[i]),:] - 3

    return bias

def init_cht_bias():
    bias = np.zeros((len(CHT_STAFF),len(CHT_ROTS)),dtype='int64') + 1

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in CHT_STAFF:
            bias[CHT_STAFF.index(LCM_STAFF[i]),:] = bias[CHT_STAFF.index(LCM_STAFF[i]),:] - 3
    return bias

def init_nuc_bias():
    bias = np.zeros((len(NUC_STAFF),len(NUC_ROTS)),dtype='int64') + 1

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in NUC_STAFF:
            bias[NUC_STAFF.index(LCM_STAFF[i]),:] = bias[NUC_STAFF.index(LCM_STAFF[i]),:] - 3
    return bias

def init_sta_bias():
    bias = np.zeros((len(STA_STAFF),len(STA_ROTS)),dtype='int64') + 1

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in STA_STAFF:
            bias[STA_STAFF.index(LCM_STAFF[i]),:] = bias[STA_STAFF.index(LCM_STAFF[i]),:] - 3
    return bias

def init_opr_bias():
    bias = np.zeros((len(OPR_STAFF),len(OPR_ROTS)),dtype='int64') + 1

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in OPR_STAFF:
            bias[OPR_STAFF.index(LCM_STAFF[i]),:] = bias[OPR_STAFF.index(LCM_STAFF[i]),:] - 3
    return bias

def init_st3_bias():
    return np.zeros((len(ST3_STAFF),len(ST3_ROTS)),dtype='int64') + 1

def init_swg_bias():
    return np.zeros((len(SWG_STAFF),len(SWG_ROTS)),dtype='int64') + 1

def init_stw_bias():
    return np.zeros((len(STW_STAFF),len(STW_ROTS)),dtype='int64') + 1

def init_wsp_bias():
    return np.zeros((len(WSP_STAFF),len(WSP_ROTS)),dtype='int64') + 1

def init_wmr_bias():
    return np.zeros((len(WMR_STAFF),len(WMR_ROTS)),dtype='int64') + 1

def init_nhk_bias():
    return np.zeros((len(NHK_STAFF),len(NHK_ROTS)),dtype='int64') + 1

def init_scv_bias():
    bias = np.zeros((len(SCV_STAFF),len(SCV_ROTS)),dtype='int64') + 1

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in SCV_STAFF:
            bias[SCV_STAFF.index(LCM_STAFF[i]),:] = bias[SCV_STAFF.index(LCM_STAFF[i]),:] - 1
    return bias

def init_adm_bias():
    bias = np.zeros((len(SCV_STAFF),len(SCV_ROTS)),dtype='int64') + 1

    return bias

def add_counter_logic(old,curr):
    '''minimum = -10 # establish a saturation point to prevent runaway values
    if old < 0 and curr > 0:
        return 1
    elif curr > 0:
        return (old+1)
    elif old == minimum-1:
        return minimum
    else:
        return old'''

    '''if old < 0 and curr > 0:
        #return int(old/2)
        return 1
    elif curr > 0:
        return (old+1)
    else:
        return old'''

    maxneg = -30

    if curr > 0:
        return 0
    elif old < maxneg:
        return maxneg+1
    return old
    
add_counter_matrix = np.vectorize(add_counter_logic)

'''
======================
 CONSTRAINT FUNCTIONS
======================
'''

def set_staffweek(cal,initials,wk,reason):

    total_slots = len(WEEK_SLOTS)+len(CALL_SLOTS)

    for j in range(total_slots):
        cal[ALL_STAFF.index(initials),j,wk] = ALL_SHIFTS.index(reason)

def set_holidays(cal,holidays):
    pass

def get_holidays(cal):
    indices = np.argwhere(cal==ALL_SHIFTS.index('Holiday'))
    holidays = np.unique(indices[:,1]/2) # divide by 2 to get the days instead of slots
    return holidays

def set_staffday(cal,initials,wk,day,reason):
    if day < len(WEEKDAYS): # block out a weekday
        if cal[ALL_STAFF.index(initials),day*2,wk] == 0: # anchors the entire day off the morning; may or may not be appropriate; prevents overwrites (Admin overwrite Vacation) 
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
        
def set_nerotation_constraints(solver,v_shifts,v_rots,v_cntr,v_rotprod_flat,v_tcost,cnts,bis,sect):

    nslts = len(WEEK_SLOTS)
    
    nstaff,nshifts,nrots,staff_tup,neshifts_tup,rots_tup = get_section_info(sect)
    #nstaff,nshifts,nrots,neshifts_tup,rots_tup = get_section_nstaff_nshifts_nrots_shifts_rots(sect)
    #_,_,staff_tup,_ = get_section_nstaff_nshifts_staff_shifts(sect)
    #print("numstaff",nstaff,"num_neshifts",nshifts,"staff_tup",staff_tup,"neshifts_tup",neshifts_tup,"num_rots",nrots,"rots_tup",rots_tup)

    # Flattened matrixes
    v_shifts_flat = [v_shifts[(stf,slt)] for stf in range(nstaff) for slt in range(nslts)]
    v_rots_flat = [v_rots[(stf,rot)] for stf in range(nstaff) for rot in range(nrots)]
    v_cntr_flat = [v_cntr[(stf,rot)] for stf in range(nstaff) for rot in range(nrots)]
    cnt_flat = [cnts[0][s,r] for s in range(nstaff) for r in range(nrots)]
    bis_flat = [bis[s,r] for s in range(nstaff) for r in range(nrots)]

    # Define the relationship between v_rots and v_shifts
    if sect == 'scv':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('SCV'))] == solver.Sum([solver.IsGreaterOrEqualCstVar(v_shifts_flat[stf*nslts+i],0) for i in range(nslts)]))

    elif sect == 'adm':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Admin'))] == solver.Sum([solver.IsGreaterOrEqualCstVar(v_shifts_flat[stf*nslts+i],0) for i in range(nslts)]))            

    # Cost function
    for i in range(nrots*nstaff):
        solver.Add(v_rotprod_flat[i] == solver.IsLessOrEqualCstVar(v_rots_flat[i],0))
        scaling_factor = 1
        solver.Add(v_cntr_flat[i] == v_rotprod_flat[i]*(int((cnt_flat[i]+bis_flat[i])/scaling_factor)))
    solver.Add(v_tcost == solver.Sum([v_cntr_flat[i] for i in range(nrots*nstaff)]))
    
    return v_tcost

def set_dayrotation_constraints(solver,v_staff,v_rots,v_cntr,v_rotprod_flat,v_pcounts,v_tcost,cnts,bis,sect,holidays):
    nslts = len(WEEK_SLOTS)
    
    nstaff,nshifts,nrots,staff_tup,shifts_tup,rots_tup = get_section_info(sect)
    #nstaff,nshifts,nrots,shifts,rots_tup = get_section_nstaff_nshifts_nrots_shifts_rots(sect)
    #_,_,staff_tup,_ = get_section_nstaff_nshifts_staff_shifts(sect)
    #print(rots_tup)
    #print(staff_tup)

    # Flattened matrixes
    v_staff_flat = [v_staff[(shf,slt)] for shf in range(nshifts) for slt in range(nslts)]
    v_rots_flat = [v_rots[(stf,rot)] for stf in range(nstaff) for rot in range(nrots)]
    v_cntr_flat = [v_cntr[(stf,rot)] for stf in range(nstaff) for rot in range(nrots)]
    cnt_flat = [cnts[0][s,r] for s in range(nstaff) for r in range(nrots)]
    #cnt_flat = [cnts[0][ALL_STAFF.index(staff_tup[s]),ALL_ROTS.index(rots_tup[r])] for s in range(nstaff) for r in range(nrots)]
    bis_flat = [bis[s,r] for s in range(nstaff) for r in range(nrots)]
    #bis_flat = [bis[ALL_STAFF.index(staff_tup[s]),ALL_ROTS.index(rots_tup[r])] for s in range(nstaff) for r in range(nrots)]
       
    # Define the relationship between v_rots and v_staff
    if sect == 'brt':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('UNC_Diag'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('UCMam Diag 8a-12p')*nslts+i],stf) for i in range(nslts)]))
            solver.Add(v_rots[(stf,rots_tup.index('UNC_Proc'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('UCMam Proc 8a-12p')*nslts+i],stf) for i in range(nslts)]))
            solver.Add(v_rots[(stf,rots_tup.index('FRE_Mamm'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('FreMam halfday')*nslts+i],stf) for i in range(nslts)]))
            solver.Add(v_rots[(stf,rots_tup.index('SLN_Mamm'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('SL Mam 8a-12p')*nslts+i],stf) for i in range(nslts)]))
            if WEEKDAYS.index('MON') in holidays:
                solver.Add(v_rots[(stf,rots_tup.index('TB'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('UCMam Proc 8a-12p')*nslts+WEEK_SLOTS.index('TUE-AM')],stf)]))
            else:
                solver.Add(v_rots[(stf,rots_tup.index('TB'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('UCMam Proc 8a-12p')*nslts],stf)]))

            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 3)

    elif sect == 'sfl':
        for stf in range(nstaff):
            slnSflIdx = [solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('SL US/Fluoro 8a-4p')*nslts+i*2],stf) for i in range(len(WEEKDAYS))] # only count every other slot (AM slot)
            freSflIdx = [solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('Fre US/Fluoro 8a-4p')*nslts+i*2],stf) for i in range(len(WEEKDAYS))]
            solver.Add(v_rots[(stf,rots_tup.index('Sonoflu'))] == solver.Sum(slnSflIdx+freSflIdx))

            slnSflIcuIdx = [solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('SL US/Fluoro 8a-4p')*nslts+WEEK_SLOTS.index('WED-AM')],stf)] # only count Wed Sonoflu ICU
            freSflIcuIdx = [solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('Fre US/Fluoro 8a-4p')*nslts+WEEK_SLOTS.index('WED-AM')],stf)]
            
            solver.Add(v_rots[(stf,rots_tup.index('Sonoflu_ICU'))] == solver.Sum(slnSflIcuIdx+freSflIcuIdx))

            #solver.Add(v_rots[(stf,rots_tup.index('FRE_Sonoflu'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('Fre US/Fluoro 8a-4p')*nslts+i],stf) for i in range(nslts)]))
            #solver.Add(v_rots[(stf,rots_tup.index('SLN_Sonoflu'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('SL US/Fluoro 8a-4p')*nslts+i],stf) for i in range(nslts)]))
            
            #solver.Add(v_rots[(stf,rots.index('Sonoflu'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(0,len(SFL_SHIFTS)*nslts,2)]))

            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 5)

        # have HG on at least 2 Sonoflu rotations per week
        # but NEED TO HANDLE CASE WHEN HG ON VACATION
        #solver.Add(v_rots[(staff_tup.index('HG'),rots.index('FRE_Sonoflu'))] > 1) # careful b/c we are counting by AM/PM shift instead of day until fixed

        #solver.Add(v_rots[(staff_tup.index('HG'),rots.index('Sonoflu'))] > 1) # careful b/c we are counting by AM/PM shift instead of day until fixed
        #solver.Add(v_rots[(staff_tup.index('HG'),rots.index('Sonoflu'))] < 5) # careful b/c we are counting by AM/PM shift instead of day until fixed

    elif sect == 'msk':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('MSK'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(nslts)])) # just need the first nslts b/c those cover the AM shift

            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 4)

    elif sect == 'ner':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Neuro'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('Neuro 8a-12p')*nslts+i],stf) for i in range(nslts)]))
            
            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 4)

    elif sect == 'abd':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Abdomen'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('Abdomen 8a-12p')*nslts+i],stf) for i in range(nslts)]))

            if WEEKDAYS.index('MON') in holidays:
                solver.Add(v_rots[(stf,rots_tup.index('Abdomen_MON'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('Abdomen 8a-12p')*nslts+WEEK_SLOTS.index('TUE-AM')],stf)]))
            else:
                solver.Add(v_rots[(stf,rots_tup.index('Abdomen_MON'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts_tup.index('Abdomen 8a-12p')*nslts],stf)]))

            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 4)
 
    elif sect == 'cht':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Chest/PET'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(nslts)])) # just need the first nslts b/c those cover the AM shift
            
            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 4)
   
    elif sect == 'nuc':
       for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Nucs'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(nslts)])) # just need the first nslts b/c only one rotation
            
            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 5)
 
    elif sect == 'sta':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('STAT_AM'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(nslts)])) # covers STAT1
            solver.Add(v_rots[(stf,rots_tup.index('STAT_PM'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(nslts,len(STA_SHIFTS)*nslts)])) # covers STAT1b and STAT2
            
            # power constraint that limits number of each rotation that staff takes
            #for rot in range(nrots):
            #    solver.Add(v_rots[(stf,rot)] < 5)
   
    elif sect == 'opr':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('OPPR_AM'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(2*nslts)])) # a hack to cover OPPR1am through OPPR2am indices
            solver.Add(v_rots[(stf,rots_tup.index('OPPR_PM'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(2*nslts,len(OPR_SHIFTS)*nslts)])) # a hack to cover OPPR3pm through OPPR4pm indices
            
            # power constraint that limits number of each rotation that staff takes
            #for rot in range(nrots):
            #    solver.Add(v_rots[(stf,rot)] < 3)

    elif sect == 'scv':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('SCV'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(len(SCV_SHIFTS)*nslts)])) 
            
            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 3)

    elif sect == 'adm':        
        # still have to work out the halfday (AM/PM) versus full day counting
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Admin'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(nslts)])) # have to figure out the correct navigation of v_staff_flat
            #solver.Add(v_rots[(0,0)] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[0],0)])) # have to figure out the correct navigation of v_staff_flat

        # Try setting certain staff_constraints
        #solver.Add(v_rots[(staff_tup.index('JDB'),rots_tup.index('Admin'))] == 1) # careful b/c we are counting by AM/PM shift instead of day until fixed
    
    else:
        pass

    # Cost function
    for i in range(nrots*nstaff):
        solver.Add(v_rotprod_flat[i] == solver.IsLessOrEqualCstVar(v_rots_flat[i],0))
        scaling_factor = 1
        solver.Add(v_cntr_flat[i] == v_rotprod_flat[i]*(int((cnt_flat[i]+bis_flat[i])/scaling_factor)))
    solver.Add(v_tcost == (solver.Sum([v_cntr_flat[i] for i in range(nrots*nstaff)])+v_pcounts))

    return v_tcost

def set_callrotation_constraints(solver,v_staff,v_rots,v_cntr,v_rotprod_flat,v_pcounts,v_tcost,cnts,bis,sect,holidays):
    nslts = len(CALL_SLOTS)
    
    #nstaff,nshifts,nrots,shifts,rots_tup = get_section_nstaff_nshifts_nrots_shifts_rots(sect)
    #_,_,staff_tup,_ = get_section_nstaff_nshifts_staff_shifts(sect)
    #print(rots_tup)
    #print(staff_tup)

    nstaff,nshifts,nrots,staff_tup,shifts_tup,rots_tup = get_section_info(sect)

    # Flattened matrixes
    v_staff_flat = [v_staff[(shf,slt)] for shf in range(nshifts) for slt in range(nslts)]
    v_rots_flat = [v_rots[(stf,rot)] for stf in range(nstaff) for rot in range(nrots)]
    v_cntr_flat = [v_cntr[(stf,rot)] for stf in range(nstaff) for rot in range(nrots)]
    cnt_flat = [cnts[0][s,r] for s in range(nstaff) for r in range(nrots)]
    bis_flat = [bis[s,r] for s in range(nstaff) for r in range(nrots)]
       
    # Define the relationship between v_rots and v_staff
    if sect == 'st3':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('STAT3'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(0,CALL_SLOTS.index('SAT-AM'))])) # just need the first nslts b/c those cover the AM shift

    elif sect == 'swg':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Swing'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(0,CALL_SLOTS.index('SAT-AM'))])) # just need the first nslts b/c those cover the AM shift

            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 2)

    elif sect == 'stw':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('STATW'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shift*nslts+i],stf) for shift in range(nshifts) for i in range(CALL_SLOTS.index('SAT-AM'),nslts)]))
            
            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 2)

    elif sect == 'wsp':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('WUSPR'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[CALL_SLOTS.index('SAT-AM')],stf)])) # just count the SAT rotation

    elif sect == 'wmr':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('WMR'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[CALL_SLOTS.index('SAT-AM')],stf)])) # just count the SAT rotation
            
    elif sect == 'nhk':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Nighthawk'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shift*nslts+i],stf) for shift in range(nshifts) for i in range(len(CALL_SLOTS))]))

    else:
        pass

    # Cost function
    for i in range(nrots*nstaff):
        solver.Add(v_rotprod_flat[i] == solver.IsLessOrEqualCstVar(v_rots_flat[i],0))
        scaling_factor = 1
        solver.Add(v_cntr_flat[i] == v_rotprod_flat[i]*(int((cnt_flat[i]+bis_flat[i])/scaling_factor)))
    solver.Add(v_tcost == (solver.Sum([v_cntr_flat[i] for i in range(nrots*nstaff)])))

    return v_tcost

def set_pooltba_constraints(solver,cal,v_shifts,v_tcost,holidays):

    for slot in range(len(WEEK_SLOTS)):
	    # it's AllDifferent except for "0" b/c this is the escape; allowed b/c shifts are assigned wrt to ALL_SHIFTS in this context
	    solver.Add(solver.AllDifferentExcept([v_shifts[(pool,slot)]*solver.IsDifferentCstVar(v_shifts[(pool,slot)],ALL_SHIFTS.index('TBA')) for pool in range(len(LCM_STAFF))],0))

    # count the number of TBA's left (will try to minimize this)
    solver.Add(v_tcost == solver.Sum([solver.IsEqualCstVar(v_shifts[(p,i)],ALL_SHIFTS.index('TBA')) for p in range(len(DBG_STAFF)) for i in range(len(WEEK_SLOTS))]))
        
    for p in range(len(DBG_STAFF)):
        staffShifts = get_staff_shifts(DBG_STAFF[p])
        dayShifts = set(DAY_SHIFTS).intersection(set(staffShifts))
        amShifts = set(AM_SHIFTS).intersection(set(staffShifts))
        pmShifts = set(PM_SHIFTS).intersection(set(staffShifts))
        print("Shifts for",DBG_STAFF[p],"AM:",amShifts,"PM:",pmShifts)

        for d in range(len(WEEKDAYS)):
            if d in holidays:
                # leave holidays unassigned
                solver.AddConstraint(v_shifts[(p,d*2)] == 0)
                solver.AddConstraint(v_shifts[(p,d*2+1)] == 0)
		#print("GOT HOLIDAY THIS WEEK:",WEEKDAYS[d])
		    
            if 'MSK 8a-12p' in dayShifts:
                AMeqMSK = solver.IsEqualCstVar(v_shifts[(p,d*2)],ALL_SHIFTS.index('MSK 8a-12p'));
                PMeqMSK = solver.IsEqualCstVar(v_shifts[(p,d*2+1)],ALL_SHIFTS.index('MSK 12-4p'));
                solver.AddConstraint(AMeqMSK == PMeqMSK)
                    
            if 'STAT1 8a-12p' in dayShifts:
                AMeqSTA = solver.IsEqualCstVar(v_shifts[(p,d*2)],ALL_SHIFTS.index('STAT1 8a-12p'));
                PMeqSTA = solver.IsEqualCstVar(v_shifts[(p,d*2+1)],ALL_SHIFTS.index('STAT1b 12p-4p'));
                solver.AddConstraint(AMeqSTA == PMeqSTA)

            if 'Abdomen 8a-12p' in dayShifts:
                AMeqABD = solver.IsEqualCstVar(v_shifts[(p,d*2)],ALL_SHIFTS.index('Abdomen 8a-12p'));
                PMeqABD = solver.IsEqualCstVar(v_shifts[(p,d*2+1)],ALL_SHIFTS.index('Abdomen 12-4p'));
                solver.AddConstraint(AMeqABD == PMeqABD)

            if 'Chest/PET 8a-12p' in dayShifts:
                AMeqCHT = solver.IsEqualCstVar(v_shifts[(p,d*2)],ALL_SHIFTS.index('Chest/PET 8a-12p'));
                PMeqCHT = solver.IsEqualCstVar(v_shifts[(p,d*2+1)],ALL_SHIFTS.index('Chest/PET 12-4p'));
                solver.AddConstraint(AMeqCHT == PMeqCHT)

            if 'Neuro 8a-12p' in dayShifts:
                AMeqNER = solver.IsEqualCstVar(v_shifts[(p,d*2)],ALL_SHIFTS.index('Neuro 8a-12p'));
                PMeqNER = solver.IsEqualCstVar(v_shifts[(p,d*2+1)],ALL_SHIFTS.index('Neuro 12-4p'));
                solver.AddConstraint(AMeqNER == PMeqNER)

            if 'SL US/Fluoro 8a-4p' in dayShifts:
                AMeqSSFL = solver.IsEqualCstVar(v_shifts[(p,d*2)],ALL_SHIFTS.index('SL US/Fluoro 8a-4p'));
                PMeqSSFL = solver.IsEqualCstVar(v_shifts[(p,d*2+1)],ALL_SHIFTS.index('SL US/Fluoro 8a-4p'));
                solver.AddConstraint(AMeqSSFL == PMeqSSFL)

            if 'Fre US/Fluoro 8a-4p' in dayShifts:
                AMeqFSFL = solver.IsEqualCstVar(v_shifts[(p,d*2)],ALL_SHIFTS.index('Fre US/Fluoro 8a-4p'));
                PMeqFSFL = solver.IsEqualCstVar(v_shifts[(p,d*2+1)],ALL_SHIFTS.index('Fre US/Fluoro 8a-4p'));
                solver.AddConstraint(AMeqFSFL == PMeqFSFL)

        for i in range(len(WEEK_SLOTS)):            
		
            if (cal[ALL_STAFF.index(DBG_STAFF[p]),i] == ALL_SHIFTS.index('TBA')):
		    if (int(i/2) in holidays):
			    solver.AddConstraint(v_shifts[(p,i)] == 0)
		    else:
			    print("Pool",DBG_STAFF[p],"set to TBA and not a holiday for slot",i)

			    # These slots need to be filled
			    # removing this makes the TBA replacement a soft cost constraint, not a hard constraint
			    #solver.AddConstraint(v_shifts[(p,i)] != 0)

			    # Only assign AM shifts to AM slots; same for PM shifts
			    if amShifts and i%2 == 0: 
				    solver.Add(solver.Max([v_shifts[(p,i)] == ALL_SHIFTS.index(r) for r in amShifts]) == 1)
			    if pmShifts and i%2 == 1:
				    print("Pool",DBG_STAFF[p],"has TBA for PM slot",i,"and must take one of their possible PM shifts",pmShifts)
				    solver.Add(solver.Max([v_shifts[(p,i)] == ALL_SHIFTS.index(r) for r in pmShifts]) == 1)

            # handle case where pool already assigned a shift from the calendar and set as a constraint; > 0 is not TBA but something else
            elif cal[ALL_STAFF.index(DBG_STAFF[p]),i] > 0:
                print("Pool",DBG_STAFF[p],"already set to take shift",ALL_SHIFTS[cal[ALL_STAFF.index(DBG_STAFF[p]),i]],"for slot",i,"index value",cal[ALL_STAFF.index(DBG_STAFF[p]),i])
                solver.Add(v_shifts[(p,i)] == cal[ALL_STAFF.index(DBG_STAFF[p]),i])
            else:
                print("Don't schedule pool",DBG_STAFF[p],"for slot",i)
                #if the person isn't assigned to work elsewhere in the same week by default block out their schedule
                    
                # may not be needed b/c biases prevent unnecessary use of pools, but may be able incorporate using the ALL_SHIFTS.index('TBA')
                #if any(w == week[0] for week in pools[p][1]): cal[ALL_STAFF.index(LCM_STAFF[p],d*2,w]) == ALL_SHIFTS.index('TBA'):
                #    cal[ALL_STAFF.index(pools[p][0]),d*2,w] = ALL_SHIFTS.index('Day Off') # AM shift
                #    cal[ALL_STAFF.index(pools[p][0]),d*2+1,w] = ALL_SHIFTS.index('Day Off') # PM shift
                solver.Add(v_shifts[(p,i)] == 0) # if not a required shift by the pool set to -1

    return v_tcost

def set_pool_constraints(solver,pools,nweeks,shifts,cal):

    for w in range(nweeks):
        for s in range(len(WEEK_SLOTS)):
            solver.Add(solver.AllDifferentExcept([shifts[(p,w,s)] for p in range(len(pools))],-1))

    for p in range(len(pools)):
        #print("ABD AM:",CCM_SHIFTS.index('MSK 8a-12p')
        staffShifts = get_staff_shifts(pools[p][0])
        dayShifts = set(DAY_SHIFTS).intersection(set(staffShifts))
        amShifts = set(AM_SHIFTS).intersection(set(staffShifts))
        pmShifts = set(PM_SHIFTS).intersection(set(staffShifts))
        #for r in amShifts: print(r)
        nshifts = len(staffShifts)
        for w in range(nweeks):
            for d in range(len(WEEKDAYS)):
                if (w,d) in pools[p][1]:                    
                    # These slots need to be filled
                    solver.AddConstraint(shifts[(p,w,d*2)] != -1)
                    solver.AddConstraint(shifts[(p,w,d*2+1)] != -1)

                    # Arrange possible assignments
                    #solver.AddConstraint(solver.AllowedAssignments(all_vars, one_day_tuples))

                    # Only assign AM shifts to AM slots; same for PM shifts
                    solver.Add(solver.Max([shifts[(p,w,d*2)] == ALL_SHIFTS.index(r) for r in amShifts]) == 1)
                    solver.Add(solver.Max([shifts[(p,w,d*2+1)] == ALL_SHIFTS.index(r) for r in pmShifts]) == 1)

                    if 'MSK 8a-12p' in dayShifts:
                        AMeqMSK = solver.IsEqualCstVar(shifts[(p,w,d*2)],ALL_SHIFTS.index('MSK 8a-12p'));
                        PMeqMSK = solver.IsEqualCstVar(shifts[(p,w,d*2+1)],ALL_SHIFTS.index('MSK 12-4p'));
                        solver.AddConstraint(AMeqMSK == PMeqMSK)
                    
                    if 'STAT1 8a-12p' in dayShifts:
                        AMeqSTA = solver.IsEqualCstVar(shifts[(p,w,d*2)],ALL_SHIFTS.index('STAT1 8a-12p'));
                        PMeqSTA = solver.IsEqualCstVar(shifts[(p,w,d*2+1)],ALL_SHIFTS.index('STAT1b 12p-4p'));
                        solver.AddConstraint(AMeqSTA == PMeqSTA)

                    if 'Abdomen 8a-12p' in dayShifts:
                        AMeqABD = solver.IsEqualCstVar(shifts[(p,w,d*2)],ALL_SHIFTS.index('Abdomen 8a-12p'));
                        PMeqABD = solver.IsEqualCstVar(shifts[(p,w,d*2+1)],ALL_SHIFTS.index('Abdomen 8a-12p'));
                        solver.AddConstraint(AMeqABD == PMeqABD)

                    if 'Chest/PET 8a-12p' in dayShifts:
                        AMeqCHT = solver.IsEqualCstVar(shifts[(p,w,d*2)],ALL_SHIFTS.index('Chest/PET 8a-12p'));
                        PMeqCHT = solver.IsEqualCstVar(shifts[(p,w,d*2+1)],ALL_SHIFTS.index('Chest/PET 12-4p'));
                        solver.AddConstraint(AMeqCHT == PMeqCHT)

                    if 'Neuro 8a-12p' in dayShifts:
                        AMeqNER = solver.IsEqualCstVar(shifts[(p,w,d*2)],ALL_SHIFTS.index('Neuro 8a-12p'));
                        PMeqNER = solver.IsEqualCstVar(shifts[(p,w,d*2+1)],ALL_SHIFTS.index('Neuro 12-4p'));
                        solver.AddConstraint(AMeqNER == PMeqNER)

                    if 'SL US/Fluoro 8a-4p' in dayShifts:
                        AMeqSSFL = solver.IsEqualCstVar(shifts[(p,w,d*2)],ALL_SHIFTS.index('SL US/Fluoro 8a-4p'));
                        PMeqSSFL = solver.IsEqualCstVar(shifts[(p,w,d*2)],ALL_SHIFTS.index('SL US/Fluoro 8a-4p'));
                        solver.AddConstraint(AMeqSSFL == PMeqSSFL)

                    if 'Fre US/Fluoro 8a-4p' in dayShifts:
                        AMeqFSFL = solver.IsEqualCstVar(shifts[(p,w,d*2)],ALL_SHIFTS.index('Fre US/Fluoro 8a-4p'));
                        PMeqFSFL = solver.IsEqualCstVar(shifts[(p,w,d*2+1)],ALL_SHIFTS.index('Fre US/Fluoro 8a-4p'));
                        solver.AddConstraint(AMeqFSFL == PMeqFSFL)
                else:
                    #if the person isn't assigned to work elsewhere in the same week by default block out their schedule
                    if any(w == week[0] for week in pools[p][1]):
                        cal[ALL_STAFF.index(pools[p][0]),d*2,w] = ALL_SHIFTS.index('Day Off') # AM shift
                        cal[ALL_STAFF.index(pools[p][0]),d*2+1,w] = ALL_SHIFTS.index('Day Off') # PM shift
                    solver.Add(shifts[(p,w,d*2)] == -1) # if not a required shift by the pool set to -1
                    solver.Add(shifts[(p,w,d*2+1)] == -1) # if not a required shift by the pool set to -1

def set_ne_calendar_constraints(solver,v_neshifts,cal,sect):
    num_slots = len(WEEK_SLOTS)

    #num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(sect)
    num_staff,num_shifts,_,staff_tup,shifts_tup,_ = get_section_info(sect)
    
    for i in range(num_staff):
        sect_allstaff_idx = ALL_STAFF.index(staff_tup[i])
        for j in range(num_slots):
            # first check if certain staff is already working during the day or on nights
            if cal[sect_allstaff_idx,j] > 0 or any([cal[sect_allstaff_idx,len(WEEK_SLOTS)+j/2] == ALL_SHIFTS.index(EVE_SHIFTS[k]) for k in range(len(EVE_SHIFTS))]):
                #if cal[sect_allstaff_idx,j] == ALL_SHIFTS.index('Vacation'):
                #    print("Staff",ALL_STAFF[sect_allstaff_idx],"is on vacation and cannot cover SCV for slot",j)
                # if that certain staff is already assigned a shift within this section, make that a constrainst in the solution
                if ALL_SHIFTS[cal[sect_allstaff_idx,j]] in shifts_tup:
                    #if sect == 'msk':
                    #    print("Staff",ALL_STAFF[ALL_STAFF.index(staff[i])],"covering shift",ALL_SHIFTS[cal[sect_allstaff_idx,j]])
                    solver.Add(v_neshifts[(i,j)] == shifts_tup.index(ALL_SHIFTS[cal[sect_allstaff_idx,j]]))
                # just make them unavailable for any of the possible section shifts
                else:
                    for k in range(num_shifts):
                        #print("Blocking Staff",staff_tup[i],"from SCV for slot",j,"shift",shifts_tup[k])
                        solver.Add(v_neshifts[(i,j)] != k)
                        
def set_day_calendar_constraints(solver,v_staff,cal,sect):
    num_slots = len(WEEK_SLOTS)

    #num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(sect)
    num_staff,num_shifts,_,staff_tup,shifts_tup,_ = get_section_info(sect)

    for i in range(num_staff):
        sect_allstaff_idx = ALL_STAFF.index(staff_tup[i])
        for j in range(num_slots):
            # first check if certain staff is already working during the day or on nights
            if cal[sect_allstaff_idx,j] > 0 or any([cal[sect_allstaff_idx,len(WEEK_SLOTS)+j/2] == ALL_SHIFTS.index(EVE_SHIFTS[k]) for k in range(len(EVE_SHIFTS))]):

                #print("Printing shift for staff",ALL_STAFF[sect_allstaff_idx],"slot",j,"shift",ALL_SHIFTS[cal[sect_allstaff_idx,j]])

                # if that certain staff is already assigned a shift within this section, make that a constrainst in the solution
                if ALL_SHIFTS[cal[sect_allstaff_idx,j]] in shifts_tup:
                    print("Setting constraint: staff",staff_tup[i],"is on",ALL_SHIFTS[cal[sect_allstaff_idx,j]],"for slot",j)
                    solver.Add(v_staff[(shifts_tup.index(ALL_SHIFTS[cal[sect_allstaff_idx,j]]),j)] == i)     

                elif ALL_SHIFTS[cal[sect_allstaff_idx,j]] == 'TBA':
                    # overwrite the TBA slot so no need to set constraint
                    #print("TBA for",ALL_STAFF[sect_allstaff_idx],"slot",j)
                    pass
                    
                # just make them unavailable for any of the possible section shifts
                else:
                    for k in range(num_shifts):
                        solver.Add(v_staff[(k,j)] != i)

def set_call_calendar_constraints(solver,stf,cal,sect,regional,week):

	num_slots = len(WEEK_SLOTS)+len(CALL_SLOTS)
	num_staff,num_shifts,_,staff_tup,shifts_tup,_ = get_section_info(sect)
	
	print("Call calendar constraints for section: ",sect)
	for i in range(num_staff):
		blocked_wknd = False
		sect_allstaff_idx = ALL_STAFF.index(staff_tup[i])

		# Handle STAT3 and Swing cases: first check if working day shifts during the week then check if already covering an evening shift
		if sect == 'st3':
			for j in range(len(WEEK_SLOTS)):
				if cal[sect_allstaff_idx,j,week] > 0:
					for k in range(num_shifts):
						solver.Add(stf[(k,int(j/2))] != i) # index the PM shift rotations
			for j in range(len(WEEKDAYS)): # iterate over the daily PM call slots
				callShift = cal[sect_allstaff_idx,len(WEEK_SLOTS)+j,week]
				if callShift > 0:
					if ALL_SHIFTS[callShift] in shifts_tup:
						solver.Add(stf[(shifts_tup.index(ALL_SHIFTS[callShift]),j)] == i)
					else:
						for k in range(num_shifts):
							solver.Add(stf[(k,j)] != i) # index the PM shift rotations

		elif sect == 'swg':
			for j in range(len(WEEK_SLOTS)):
				if cal[sect_allstaff_idx,j,week] > 0 and (ALL_SHIFTS[cal[sect_allstaff_idx,j,week]] in NOSWING):
					for k in range(num_shifts):
						print("leave Swing constraint:",k,int(j/2),"for staff",staff_tup[i])                        
						solver.Add(stf[(k,int(j/2))] != i) # index the PM shift rotations
			for j in range(len(WEEKDAYS)): # iterate over the daily PM call slots
				callShift = cal[sect_allstaff_idx,len(WEEK_SLOTS)+j,week]
				if callShift > 0:
					if ALL_SHIFTS[callShift] in shifts_tup:
						solver.Add(v_staff[(shifts_tup.index(ALL_SHIFTS[callShift]),j)] == i)
					else:
						for k in range(num_shifts):
							solver.Add(stf[(k,j)] != i) # index the PM shift rotations

		elif sect == 'nhk':
			for j in range(len(WEEK_SLOTS)):
				if cal[sect_allstaff_idx,j,week] > 0:
					for k in range(num_shifts):
						solver.Add(stf[(k,int(j/2))] != i) # index the PM shift rotations
			for j in range(len(WEEKDAYS)): # iterate over the daily PM call slots
				callShift = cal[sect_allstaff_idx,len(WEEK_SLOTS)+j,week]
				if callShift > 0:
					if ALL_SHIFTS[callShift] in shifts_tup:
						solver.Add(stf[(shifts_tup.index(ALL_SHIFTS[callShift]),j)] == i)
					else:
						for k in range(num_shifts):
							solver.Add(stf[(k,j)] != i) # index the PM shift rotations
						
		else: # we are dealing with weekend rotation
			for j in range(len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM'),len(WEEK_SLOTS)+len(CALL_SLOTS)):
				callShift = cal[sect_allstaff_idx,j,week]
				if callShift > 0:
					if ALL_SHIFTS[callShift] in shifts_tup:
						solver.Add(v_staff[(shifts_tup.index(ALL_SHIFTS[callShift]),j-len(WEEK_SLOTS))] == i) # j is wrt the calendar so need to subtract WEEK_SLOTS to get back in the CALL_SLOT context
					blocked_wknd = True
			if blocked_wknd and sect in WKND_SECTS:
				for j in range(CALL_SLOTS.index('SAT-AM'),len(CALL_SLOTS)):
					for k in range(num_shifts):
						#print("leave STATW constraint:",k,j,staff_tup[i])                        
						solver.Add(stf[(k,j)] != i)

def set_brt_constraints(s,st,cal,holidays): # s = solver

    for i in range(len(WEEK_SLOTS)):
                
        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(BRT_SHIFTS))],-1))

    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2)] == st[(BRT_SHIFTS.index('UCMam Diag 12-4p'),i*2+1)])
        s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2)] == st[(BRT_SHIFTS.index('UCMam Proc 12-4p'),i*2+1)])
        
        # CCM doesn't go to UNC
        s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2)] != BRT_STAFF.index('CCM'))
        s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2)] != BRT_STAFF.index('CCM'))
        s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),i*2)] != BRT_STAFF.index('CCM'))
      
        # Shifts that don't fit into context (e.g. UCMam Diag 12-4p on a morning shift)
        s.Add(st[(BRT_SHIFTS.index('UCMam Diag 12-4p'),i*2)] == -1)
        s.Add(st[(BRT_SHIFTS.index('UCMam Proc 12-4p'),i*2)] == -1)
        s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2+1)] == -1)
        s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2+1)] == -1)

    # Shifts that don't fit into context
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),0)] == -1)
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),1)] == -1)
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),3)] == -1)
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),4)] == -1)
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),5)] == -1)
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),7)] == -1)
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),8)] == -1)
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),9)] == -1)
    
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),1)] == -1)
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),2)] == -1)
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),5)] == -1)
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),6)] == -1)
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),9)] == -1)
            
    for i in range(len(WEEKDAYS)):
            
        # Real shifts unless it's a holiday
        if i not in holidays:
            s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2)] != -1)
            s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2)] != -1)
            s.Add(st[(BRT_SHIFTS.index('UCMam Diag 12-4p'),i*2+1)] != -1)
            s.Add(st[(BRT_SHIFTS.index('UCMam Proc 12-4p'),i*2+1)] != -1)
        else: 
            s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2)] == -1)
            s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2)] == -1)
            s.Add(st[(BRT_SHIFTS.index('UCMam Diag 12-4p'),i*2+1)] == -1)
            s.Add(st[(BRT_SHIFTS.index('UCMam Proc 12-4p'),i*2+1)] == -1)

    # Handle the FRE/SLN rotations with and without holidays
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),2)] != -1) if (1 not in holidays) else s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),2)] == -1)
    s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),6)] != -1) if (3 not in holidays) else s.Add(st[(BRT_SHIFTS.index('SL Mam 8a-12p'),6)] == -1)
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),0)] != -1) if (0 not in holidays) else s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),0)] == -1)
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),3)] != -1) if (1 not in holidays) else s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),3)] == -1)
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),4)] != -1) if (2 not in holidays) else s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),4)] == -1) 
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),7)] != -1) if (3 not in holidays) else s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),7)] == -1)
    s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),8)] != -1) if (4 not in holidays) else s.Add(st[(BRT_SHIFTS.index('FreMam halfday'),8)] == -1)
    
    # Don't be on the same UNC rotation two days in a row (can relax if short-staffed)
    #if i < 4:
    #    s.Add(st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2)] != st[(BRT_SHIFTS.index('UCMam Proc 8a-12p'),i*2+2)])
    #    s.Add(st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2)] != st[(BRT_SHIFTS.index('UCMam Diag 8a-12p'),i*2+2)])

def set_sfl_constraints(s,st,cal,holidays): # s = solver
    
    # Don't cover the same Sonoflu shift in 1 week; this was originally put in in but not sure I understand purpose
    #s.Add(s.AllDifferent([st[(j*2,i*2)] for j in range(len(SFL_SHIFTS)/2) for i in range(len(WEEKDAYS))]))

    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(SFL_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] == st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2+1)])
        s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] == st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2+1)])

        # These shifts are real and need to be assigned unless it's a holiday
        if i not in holidays:
            s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != -1)
            s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] != -1)
            s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2+1)] != -1)
            s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2+1)] != -1)
        else:
            s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] == -1)
            s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] == -1)
            s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2+1)] == -1)
            s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2+1)] == -1)

        # Don't be on Sonoflu 2 days in a row
        if i < 4:
            # for same location
            s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2+2)])
            s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] != st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2+2)])
            
            # for different location
            s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] != st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2+2)])
            s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2+2)])

        # HG doesn't go to SLN
        s.Add(st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),i*2)] != SFL_STAFF.index('HG'))

        # DRL, JFK, SMN don't go to FRE
        s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != SFL_STAFF.index('JFK'))
        s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != SFL_STAFF.index('SDE'))
        s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != SFL_STAFF.index('DRL'))
        s.Add(st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),i*2)] != SFL_STAFF.index('SMN'))

    # Only MSK person can cover SLN TUE unless it's a holiday then doesn't matter
    if WEEKDAYS.index('TUE') not in holidays: s.Add(s.Max([st[(SFL_SHIFTS.index('SL US/Fluoro 8a-4p'),WEEK_SLOTS.index('TUE-AM'))] == SFL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)

    # Only MSK person can cover FRE THU unless it's a holiday then doesn't matter
    if WEEKDAYS.index('THU') not in holidays: s.Add(s.Max([st[(SFL_SHIFTS.index('Fre US/Fluoro 8a-4p'),WEEK_SLOTS.index('THU-AM'))] == SFL_STAFF.index(rad) for rad in MSK_STAFF]) == 1)

def set_msk_constraints(s,st,cal,holidays): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(MSK_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(MSK_SHIFTS.index('MSK 8a-12p'),i*2)] == st[(MSK_SHIFTS.index('MSK 12-4p'),i*2+1)])
        
        # These shifts are real and need to be assigned unless a holiday
        if i not in holidays:
            s.Add(st[(MSK_SHIFTS.index('MSK 8a-12p'),i*2)] != -1)
            s.Add(st[(MSK_SHIFTS.index('MSK 12-4p'),i*2+1)] != -1)
        else:
            s.Add(st[(MSK_SHIFTS.index('MSK 8a-12p'),i*2)] == -1)
            s.Add(st[(MSK_SHIFTS.index('MSK 12-4p'),i*2+1)] == -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(MSK_SHIFTS.index('MSK 12-4p'),i*2)] == -1)
        s.Add(st[(MSK_SHIFTS.index('MSK 8a-12p'),i*2+1)] == -1)

def set_abd_constraints(s,st,cal,holidays): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(ABD_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(ABD_SHIFTS.index('Abdomen 8a-12p'),i*2)] == st[(ABD_SHIFTS.index('Abdomen 12-4p'),i*2+1)])

        # These shifts are real and need to be assigned unless a holiday
        if i not in holidays:
            s.Add(st[(ABD_SHIFTS.index('Abdomen 8a-12p'),i*2)] != -1)
            s.Add(st[(ABD_SHIFTS.index('Abdomen 12-4p'),i*2+1)] != -1)
        else:
            s.Add(st[(ABD_SHIFTS.index('Abdomen 8a-12p'),i*2)] == -1)
            s.Add(st[(ABD_SHIFTS.index('Abdomen 12-4p'),i*2+1)] == -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(ABD_SHIFTS.index('Abdomen 12-4p'),i*2)] == -1)
        s.Add(st[(ABD_SHIFTS.index('Abdomen 8a-12p'),i*2+1)] == -1)

def set_ner_constraints(s,st,cal,holidays): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(NER_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(NER_SHIFTS.index('Neuro 8a-12p'),i*2)] == st[(NER_SHIFTS.index('Neuro 12-4p'),i*2+1)])

        # These shifts are real and need to be assigned unless a holiday
        if i not in holidays:
            s.Add(st[(NER_SHIFTS.index('Neuro 8a-12p'),i*2)] != -1)
            s.Add(st[(NER_SHIFTS.index('Neuro 12-4p'),i*2+1)] != -1)
        else:
            s.Add(st[(NER_SHIFTS.index('Neuro 8a-12p'),i*2)] == -1)
            s.Add(st[(NER_SHIFTS.index('Neuro 12-4p'),i*2+1)] == -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(NER_SHIFTS.index('Neuro 12-4p'),i*2)] == -1)
        s.Add(st[(NER_SHIFTS.index('Neuro 8a-12p'),i*2+1)] == -1)

def set_cht_constraints(s,st,cal,holidays): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(CHT_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(CHT_SHIFTS.index('Chest/PET 8a-12p'),i*2)] == st[(CHT_SHIFTS.index('Chest/PET 12-4p'),i*2+1)])

        # These shifts are real and need to be assigned unless a holiday
        if i not in holidays:
            s.Add(st[(CHT_SHIFTS.index('Chest/PET 8a-12p'),i*2)] != -1)
            s.Add(st[(CHT_SHIFTS.index('Chest/PET 12-4p'),i*2+1)] != -1)
        else:
            s.Add(st[(CHT_SHIFTS.index('Chest/PET 8a-12p'),i*2)] == -1)
            s.Add(st[(CHT_SHIFTS.index('Chest/PET 12-4p'),i*2+1)] == -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(CHT_SHIFTS.index('Chest/PET 12-4p'),i*2)] == -1)
        s.Add(st[(CHT_SHIFTS.index('Chest/PET 8a-12p'),i*2+1)] == -1)

def set_nuc_constraints(s,st,cal,holidays): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(NUC_SHIFTS))],-1)) # ? whether this is necessary; should revisit
        
    for i in range(len(WEEKDAYS)):

        # Shifts that don't fit into context (e.g. Nucs not an AM shift)
        s.Add(st[(NUC_SHIFTS.index('Nucs 8a-4p'),i*2)] == -1)

        # The PM Nucs shift must be filled unless it's a holiday
        if i not in holidays:
            s.Add(st[(NUC_SHIFTS.index('Nucs 8a-4p'),i*2+1)] != -1)
        else:
            s.Add(st[(NUC_SHIFTS.index('Nucs 8a-4p'),i*2+1)] == -1)

def set_sta_constraints(s,st,cal,holidays): # s = solver
    
    for i in range(len(WEEK_SLOTS)):

        # More than 1 staff would never cover 1 assignment
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(STA_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2)] == st[(STA_SHIFTS.index('STAT1b 12p-4p'),i*2+1)])

        # If not a holiday, these shifts are real and need to be assigned unless it's a holiday
        if i not in holidays:
            s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2)] != -1)
            s.Add(st[(STA_SHIFTS.index('STAT1b 12p-4p'),i*2+1)] != -1)
            s.Add(st[(STA_SHIFTS.index('STAT2 12p-4p'),i*2+1)] != -1)
        else:
            s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2)] == -1)
            s.Add(st[(STA_SHIFTS.index('STAT1b 12p-4p'),i*2+1)] == -1)
            s.Add(st[(STA_SHIFTS.index('STAT2 12p-4p'),i*2+1)] == -1)

        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(STA_SHIFTS.index('STAT1b 12p-4p'),i*2)] == -1)
        s.Add(st[(STA_SHIFTS.index('STAT2 12p-4p'),i*2)] == -1)
        s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2+1)] == -1)

        # Don't be on all day STAT two days in a row 
  
	# relax this constraint for now
        #if i < 4:
        #    s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2)] != st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2+2)])

def set_opr_constraints(s,st,cal,holidays): # s = solver
    
	# Handle special cases for HG
	hg_idx = ALL_STAFF.index('HG')

	for i in range(len(WEEK_SLOTS)):

		# No double coverage
		s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(OPR_SHIFTS))],-1))

		# Note: this may be better handled elsewehere or in another way; may cause problems if pools take up the OPPR shifts beforehand and nothing is left for HG
		# If HG works Nucs in the afternoon, put OPPR in the AM
		'''if cal[hg_idx,i] == 0: # if HG not scheduled 
		if i%2 == 1: # PM case
                s.Add(s.Max([st[(k,i)] == hg_idx for k in range(2,4)]) == 1) # the range specificies the PM OPPR
		else: # AM case
                s.Add(s.Max([st[(k,i)] == hg_idx for k in range(0,2)]) == 1)''' # the range specificies the AM OPPR
		
	for i in range(len(WEEKDAYS)):

	    # These shifts are real and need to be assigned unless it's a holiday
	    if i not in holidays:
		s.Add(st[(OPR_SHIFTS.index('OPPR1am'),i*2)] != -1)
		s.Add(st[(OPR_SHIFTS.index('OPPR2am'),i*2)] != -1)
		s.Add(st[(OPR_SHIFTS.index('OPPR3pm'),i*2+1)] != -1)
		s.Add(st[(OPR_SHIFTS.index('OPPR4pm'),i*2+1)] != -1)
	    else:
		s.Add(st[(OPR_SHIFTS.index('OPPR1am'),i*2)] == -1)
		s.Add(st[(OPR_SHIFTS.index('OPPR2am'),i*2)] == -1)
		s.Add(st[(OPR_SHIFTS.index('OPPR3pm'),i*2+1)] == -1)
		s.Add(st[(OPR_SHIFTS.index('OPPR4pm'),i*2+1)] == -1)
		
	    # Shifts that don't fit into context (e.g. PM on a morning shift)
	    s.Add(st[(OPR_SHIFTS.index('OPPR3pm'),i*2)] == -1)
	    s.Add(st[(OPR_SHIFTS.index('OPPR4pm'),i*2)] == -1)
	    s.Add(st[(OPR_SHIFTS.index('OPPR1am'),i*2+1)] == -1)
	    s.Add(st[(OPR_SHIFTS.index('OPPR2am'),i*2+1)] == -1)

def set_st3_constraints(s,st,cal,holidays,regional,week): # s = solver

	# STAT3 person is for the whole week
	for i in range(len(CALL_SLOTS)-5): # subtract the weekend days to get MON-THU (the last statement will be THU == FRI, that's why only 'til THU)
		s.Add(st[(ST3_SHIFTS.index('STAT3 4p-11p'),i)] == st[(ST3_SHIFTS.index('STAT3 4p-11p'),i+1)])
            
	for i in range(len(CALL_SLOTS)):
			
		if i < CALL_SLOTS.index('SAT-AM'): 
			# These shifts are real and need to be assigned (MON-FRI STAT3); figure out how to handle the holiday
			s.Add(st[(ST3_SHIFTS.index('STAT3 4p-11p'),i)] != -1)
		else:
			# Shifts that don't fit into context (e.g. STAT3 not on weekends)
			s.Add(st[(ST3_SHIFTS.index('STAT3 4p-11p'),i)] == -1)

	# Handle 4pm-12am regional stroke shifts
	if regional:
		strokeShifts = [regional[j][1:] for j in range(len(regional)) if regional[j][0] == 'Regional Stroke Alert 4p-12a']

	print("Stroke Shifts (weekday eve):",strokeShifts)
	
	for i in range(len(CALL_SLOTS)):
		if i < CALL_SLOTS.index('SAT-AM'):

			'''print("Stroke list length:",len(strokeShifts[0]))
			for j in range(len(strokeShifts[0])):
				print("strokeShift[0][j][0]",int(strokeShifts[0][j][0]-1),"week",week)
				print("strokeShift[0][j][1]",strokeShifts[0][j][1])'''
			
			if [strokeShifts[0][j] for j in range(len(strokeShifts[0])) if ((strokeShifts[0][j][0]-1 == week) and (strokeShifts[0][j][1] == CALL_SLOTS[i]))]:
				#print("Setting regional stroke constraint for",CALL_SLOTS[i])
				s.Add(s.Max([st[(ST3_SHIFTS.index('Regional Stroke Alert 4p-12a'),i)] == ST3_STAFF.index(rad) for rad in NER_STAFF if rad in ST3_STAFF]) == 1)
			else:
				#print("Blocking regional stroke alert ST3 shift for",CALL_SLOTS[i])
				s.Add(st[(ST3_SHIFTS.index('Regional Stroke Alert 4p-12a'),i)] == -1)
		else:
			s.Add(st[(ST3_SHIFTS.index('Regional Stroke Alert 4p-12a'),i)] == -1)

	# No one is on stroke alert for more than once per week
	s.Add(s.AllDifferentExcept([st[(ST3_SHIFTS.index('Regional Stroke Alert 4p-12a'),i)] for i in range(len(CALL_SLOTS))],-1))

def set_swg_constraints(s,st,cal,holidays): # s = solver
	
	# Only one Swing shift per week
	s.Add(s.AllDifferentExcept([st[(j,i)] for i in range(len(CALL_SLOTS)) for j in range(len(SWG_SHIFTS))],-1))

	for i in range(len(CALL_SLOTS)):
		
		if i < CALL_SLOTS.index('SAT-AM'): 
			# These shifts are real and need to be assigned (MON-FRI SWG)
			s.Add(st[(SWG_SHIFTS.index('Swing'),i)] != -1)
		else:
			# Shifts that don't fit into context (e.g. SWG not on weekends)
			s.Add(st[(SWG_SHIFTS.index('Swing'),i)] == -1)

def set_stw_constraints(s,st,cal,holidays,regional,week): # s = solver
    
	# Only one STAT shift per weekend
	s.Add(s.AllDifferentExcept([st[(j,i)] for i in range(len(CALL_SLOTS)) for j in range(len(STW_SHIFTS))],-1))
        
	for i in range(len(CALL_SLOTS)):
		if i < CALL_SLOTS.index('SAT-AM'):             
			# Shifts that don't fit into context (e.g. STATW on weekdays)
			s.Add(st[(STW_SHIFTS.index('STATWAM 8a-330p'),i)] == -1)
			s.Add(st[(STW_SHIFTS.index('STATWPM 330p-11p'),i)] == -1)
		elif (i == CALL_SLOTS.index('SAT-AM')) or (i == CALL_SLOTS.index('SUN-AM')):
			s.Add(st[(STW_SHIFTS.index('STATWAM 8a-330p'),i)] != -1)
			s.Add(st[(STW_SHIFTS.index('STATWPM 330p-11p'),i)] == -1)
		elif (i == CALL_SLOTS.index('SAT-PM')) or (i == CALL_SLOTS.index('SUN-PM')):
			s.Add(st[(STW_SHIFTS.index('STATWAM 8a-330p'),i)] == -1)
			s.Add(st[(STW_SHIFTS.index('STATWPM 330p-11p'),i)] != -1)
		else:
			pass
			#s.Add(st[(STW_SHIFTS.index('STATWAM 8a-330p'),i)] == -1)
			#s.Add(st[(STW_SHIFTS.index('STATWPM 330p-11p'),i)] != -1)
		
		
	# Handle weekend regional stroke shifts
	if regional:
		strokeAMShifts = [regional[j][1:] for j in range(len(regional)) if regional[j][0] == 'Regional Stroke Alert 8a-4p']
		strokePMShifts = [regional[j][1:] for j in range(len(regional)) if regional[j][0] == 'Regional Stroke Alert 4p-12a']
		
	#print("Stroke AM Shifts (weekend constraint):",strokeAMShifts[0])
	#print("Stroke PM Shifts (weekend constraint):",strokePMShifts[0])
	
	for i in range(len(CALL_SLOTS)):
		if (i == CALL_SLOTS.index('SAT-AM')) or (i == CALL_SLOTS.index('SUN-AM')):
			# PM shift does not apply
			s.Add(st[(STW_SHIFTS.index('Regional Stroke Alert 4p-12a'),i)] == -1)

			#for j in range(len(strokeAMShifts[0])):
				#print("strokeAMShift[0][j][0]",int(strokeAMShifts[0][j][0]-1),"week",week)
				#print("strokeAMShift[0][j][1]",strokeAMShifts[0][j][1])
			
			if [strokeAMShifts[0][j] for j in range(len(strokeAMShifts[0])) if (strokeAMShifts[0][j][0]-1 == week) and (strokeAMShifts[0][j][1] == CALL_SLOTS[i])]:
				s.Add(s.Max([st[(STW_SHIFTS.index('Regional Stroke Alert 8a-4p'),i)] == STW_STAFF.index(rad) for rad in NER_STAFF if rad in STW_STAFF]) == 1)
			else:
				s.Add(st[(STW_SHIFTS.index('Regional Stroke Alert 8a-4p'),i)] == -1)

		elif (i == CALL_SLOTS.index('SAT-PM')) or (i == CALL_SLOTS.index('SUN-PM')):
			# AM shift does not apply
			s.Add(st[(STW_SHIFTS.index('Regional Stroke Alert 8a-4p'),i)] == -1)

			if [strokePMShifts[0][j] for j in range(len(strokePMShifts[0])) if (strokePMShifts[0][j][0]-1 == week) and (strokePMShifts[0][j][1] == CALL_SLOTS[i])]:
				s.Add(s.Max([st[(STW_SHIFTS.index('Regional Stroke Alert 4p-12a'),i)] == STW_STAFF.index(rad) for rad in NER_STAFF if rad in STW_STAFF]) == 1)
			else:
				s.Add(st[(STW_SHIFTS.index('Regional Stroke Alert 4p-12a'),i)] == -1)
		else:
			s.Add(st[(STW_SHIFTS.index('Regional Stroke Alert 8a-4p'),i)] == -1)
			s.Add(st[(STW_SHIFTS.index('Regional Stroke Alert 4p-12a'),i)] == -1)


def set_wsp_constraints(s,st,cal,holidays): # s = solver
    
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

def set_wmr_constraints(s,st,cal,holidays): # s = solver
    
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

def set_nhk_constraints(s,st,cal,holidays,regional,week):

	if regional:
		genNhkWks = [regional[j][1:] for j in range(len(regional)) if regional[j][0] == 'Nightshift 11p-12a']
		neuroNhkWks = [regional[j][1:] for j in range(len(regional)) if regional[j][0] == 'NeuroNH 11p-12a']

	print("Neuro NHK Weeks:",neuroNhkWks)
	print("General NHK Weeks:",genNhkWks)
	
	for shift in NHK_SHIFTS:

		# Local variables
		currWk = False
		priorWk = False

		if shift == 'NeuroNH 11p-12a':
			wklist = neuroNhkWks
		elif shift == 'Nightshift 11p-12a':
			wklist = genNhkWks

		if wklist:
			print("wklist[0]",wklist[0])
			
			for i in wklist[0]:

				# all constraints for NHK should be based on scheduling from the previous week; the set_call_calendar_constraints function will take care of the weekday scheduling
				if week == i-1:
					currWk = True
					for k in range(len(CALL_SLOTS)):
						if (k == CALL_SLOTS.index('FRI-PM')) or (k == CALL_SLOTS.index('SAT-PM')) or (k == CALL_SLOTS.index('SUN-PM')):
							# Real shifts
							if shift == 'NeuroNH 11p-12a': 
								s.Add(s.Max([st[(NHK_SHIFTS.index(shift),k)] == NHK_STAFF.index(rad) for rad in NER_STAFF if rad in NHK_STAFF]) == 1)
							elif shift == 'Nightshift 11p-12a':
								s.Add(s.Max([st[(NHK_SHIFTS.index(shift),k)] == NHK_STAFF.index(rad) for rad in NHK_STAFF if rad not in NER_STAFF]) == 1)

					s.Add(st[(NHK_SHIFTS.index(shift),CALL_SLOTS.index('FRI-PM'))] == st[(NHK_SHIFTS.index(shift),CALL_SLOTS.index('SAT-PM'))])
					s.Add(st[(NHK_SHIFTS.index(shift),CALL_SLOTS.index('SAT-PM'))] == st[(NHK_SHIFTS.index(shift),CALL_SLOTS.index('SUN-PM'))]) 

				# if this is not a week in which a nighthawk shift starts, don't schedule unless already done so
				elif week == i:
					priorWk = True
					for k in range(0,CALL_SLOTS.index('FRI-PM')):
						# Real shifts
						if shift == 'NeuroNH 11p-12a': 
							s.Add(s.Max([st[(NHK_SHIFTS.index(shift),k)] == NHK_STAFF.index(rad) for rad in NER_STAFF if rad in NHK_STAFF]) == 1)
						elif shift == 'Nightshift 11p-12a':
							s.Add(s.Max([st[(NHK_SHIFTS.index(shift),k)] == NHK_STAFF.index(rad) for rad in NHK_STAFF if rad not in NER_STAFF]) == 1)
							
					for j in range(0,CALL_SLOTS.index('THU-PM')): # the last loop will set WED = THU
						s.Add(st[(NHK_SHIFTS.index(shift),j)] == st[(NHK_SHIFTS.index(shift),j+1)])

		# Overarching constraints to specify shifts that don't need to be filled
		for k in range(len(CALL_SLOTS)):
			if (k == CALL_SLOTS.index('SAT-AM')) or (k == CALL_SLOTS.index('SUN-AM')): 
				# These shifts are not real
				s.Add(st[(NHK_SHIFTS.index(shift),k)] == -1)

			if not currWk and not priorWk:
				s.Add(st[(NHK_SHIFTS.index(shift),k)] == -1)

			elif currWk and not priorWk:
				if k < CALL_SLOTS.index('FRI-PM'):
					s.Add(st[(NHK_SHIFTS.index(shift),k)] == -1)

			elif priorWk and not currWk:
				if k > CALL_SLOTS.index('THU-PM'):
					s.Add(st[(NHK_SHIFTS.index(shift),k)] == -1)

def set_scv_constraints(s,st,cal,holidays): # s = solver

    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(SCV_SHIFTS))],-1))

    # On Mondays set having an NEU, MSK, and ABD/CHT SCV unless it's a holiday 
    if WEEKDAYS.index('MON') not in holidays: s.Add(s.Max([st[(SCV_SHIFTS.index('SCV AM'),WEEK_SLOTS.index('MON-AM'))] == SCV_STAFF.index(rad) for rad in NER_STAFF]) == 1)
    if WEEKDAYS.index('MON') not in holidays: s.Add(s.Max([st[(SCV_SHIFTS.index('SCV2 AM'),WEEK_SLOTS.index('MON-AM'))] == SCV_STAFF.index(rad) for rad in MSK_STAFF+ABD_STAFF]) == 1)
    #s.Add(s.Max([st[(SCV_SHIFTS.index('SCV3 AM'),WEEK_SLOTS.index('MON-AM'))] == SCV_STAFF.index(rad) for rad in ABD_STAFF]) == 1)
    
    for i in range(len(WEEKDAYS)):
        
        # Shifts that don't fit into context (e.g. PM on a morning shift)
        s.Add(st[(SCV_SHIFTS.index('SCV PM'),i*2)] == -1)
        s.Add(st[(SCV_SHIFTS.index('SCV2 PM'),i*2)] == -1)
        s.Add(st[(SCV_SHIFTS.index('SCV AM'),i*2+1)] == -1)
        s.Add(st[(SCV_SHIFTS.index('SCV2 AM'),i*2+1)] == -1)
        s.Add(st[(SCV_SHIFTS.index('SCV3 AM'),i*2+1)] == -1)

        # don't assign real SCV shifts if it's a holiday 
        if i in holidays:
            s.Add(st[(SCV_SHIFTS.index('SCV PM'),i*2+1)] == -1)
            s.Add(st[(SCV_SHIFTS.index('SCV2 PM'),i*2+1)] == -1)
            s.Add(st[(SCV_SHIFTS.index('SCV AM'),i*2)] == -1)
            s.Add(st[(SCV_SHIFTS.index('SCV2 AM'),i*2)] == -1)
            s.Add(st[(SCV_SHIFTS.index('SCV3 AM'),i*2)] == -1)
   
def set_ne_constraints(solver,v_shifts,cal,holidays,section):

    num_staff,num_shifts,num_rots,staff_tup,shifts_tup,rots_tup = get_section_info(section)
    #num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(section)
    #_,_,num_rots,_,rots_tup = get_section_nstaff_nshifts_nrots_shifts_rots(section)

    if section == 'adm':
        for s in range(num_staff):
            for i in range(len(WEEKDAYS)):

                # Shifts that don't fit into context (e.g. PM on a morning shift)
                solver.Add(v_shifts[(s,i*2)] != ADM_SHIFTS.index('Admin PM'))
                solver.Add(v_shifts[(s,i*2+1)] != ADM_SHIFTS.index('Admin AM'))

                # Link the AM/PM rotations for the Admin Day assignments
                AMeqAdmin = solver.IsEqualCstVar(v_shifts[(s,i*2)],ADM_SHIFTS.index('Admin Day'));
                PMeqAdmin = solver.IsEqualCstVar(v_shifts[(s,i*2+1)],ADM_SHIFTS.index('Admin Day'));
                solver.AddConstraint(AMeqAdmin == PMeqAdmin)

                if i in holidays:
                    solver.Add(v_shifts[(s,i*2)] == -1)
                    solver.Add(v_shifts[(s,i*2+1)] == -1)
                else:
                    pass
                    # probably conflicts with imports where Admin Day has already been assigned
                    #solver.Add(v_shifts[(s,i*2)] != 0) # for now don't schedule full admin days unless previously requested

        solver.Add(solver.Sum([solver.IsGreaterOrEqualCstVar(v_shifts[(s,j)],0) for s in range(num_staff) for j in range(len(WEEK_SLOTS))]) < 10)

    elif section == 'scv':
        for s in range(num_staff):
            for i in range(len(WEEKDAYS)):

                # Shifts that don't fit into context (e.g. PM on a morning shift)
                solver.Add(v_shifts[(s,i*2)] != SCV_SHIFTS.index('SCV PM'))
                solver.Add(v_shifts[(s,i*2+1)] != SCV_SHIFTS.index('SCV AM'))

                # Don't put people on for full days of SCV
                solver.Add(solver.Sum([solver.IsGreaterOrEqualCstVar(v_shifts[(s,i*2+j)],0) for j in range(2)]) < 2)

                # fails b/c of -1 case
                '''AMeqSCV = solver.IsEqualCstVar(v_shifts[(s,i*2)],SCV_SHIFTS.index('SCV AM'))
                PMeqSCV = solver.IsEqualCstVar(v_shifts[(s,i*2+1)],SCV_SHIFTS.index('SCV PM'))
                solver.AddConstraint(AMeqSCV == PMeqSCV)'''

                if i in holidays:
                    solver.Add(v_shifts[(s,i*2)] == -1)
                    solver.Add(v_shifts[(s,i*2+1)] == -1)

        # MSK, Neuro, and Abdominal SCV on Mondays; in other words we should have > 0 from each and a total of 3; each == 1 doesn't work b/c certain staff overlap with MSK/NER
        if WEEKDAYS.index('MON') not in holidays:
            solver.Add(solver.Sum([solver.IsEqualCstVar(v_shifts[(s,WEEK_SLOTS.index('MON-AM'))],0) for s in range(num_staff) if staff_tup[s] in MSK_STAFF]) == 1)
            solver.Add(solver.Sum([solver.IsEqualCstVar(v_shifts[(s,WEEK_SLOTS.index('MON-AM'))],0) for s in range(num_staff) if staff_tup[s] in NER_STAFF]) == 1)
            solver.Add(solver.Sum([solver.IsEqualCstVar(v_shifts[(s,WEEK_SLOTS.index('MON-AM'))],0) for s in range(num_staff) if staff_tup[s] in ABD_STAFF]) == 1)
            solver.Add(solver.Sum([solver.IsEqualCstVar(v_shifts[(s,WEEK_SLOTS.index('MON-AM'))],0) for s in range(num_staff)]) == 3)
        else: # if there is a Monday holiday, then allocate all the SCV rotations for Tuesday
            #pass
            #solver.Add(solver.Sum([solver.IsEqualCstVar(v_shifts[(s,WEEK_SLOTS.index('TUE-AM'))],0) for s in range(num_staff) if staff_tup[s] in MSK_STAFF]) == 1)
            solver.Add(solver.Sum([solver.IsEqualCstVar(v_shifts[(s,WEEK_SLOTS.index('TUE-AM'))],0) for s in range(num_staff) if staff_tup[s] in NER_STAFF]) == 1)
            solver.Add(solver.Sum([solver.IsEqualCstVar(v_shifts[(s,WEEK_SLOTS.index('TUE-AM'))],0) for s in range(num_staff) if staff_tup[s] in ABD_STAFF]) == 1)
            #solver.Add(solver.Sum([solver.IsEqualCstVar(v_shifts[(s,WEEK_SLOTS.index('TUE-AM'))],0) for s in range(num_staff)]) == 3)

        # limit total number of SCV half day shifts in the week;
        solver.Add(solver.Sum([solver.IsGreaterOrEqualCstVar(v_shifts[(s,j)],0) for s in range(num_staff) for j in range(len(WEEK_SLOTS))]) < 10)

        # limit the number of SCV shifts per day
        for i in range(len(WEEKDAYS)):
            solver.Add(solver.Sum([solver.IsGreaterOrEqualCstVar(v_shifts[(s,i*2+j)],0) for s in range(num_staff) for j in range(2)]) < 4)

    else:
        pass

# copies the constraint driven pool calendar (uses pools only when needed) to a new calendar that will be filled in
def set_pooltba_cal(tbacal,oldcal,nweeks):
    for w in range(nweeks):
        for i in range(len(LCM_STAFF)):
            for j in range(len(WEEK_SLOTS)):
                tbacal[ALL_STAFF.index(LCM_STAFF[i]),j,w] = oldcal[ALL_STAFF.index(LCM_STAFF[i]),j,w]
             
'''
====================
 ANALYSIS FUNCTIONS
====================
'''

def update_allcounter(r_cumulative,r_counter,cumulative,counter,section):

    #nstaff,nrots,staff_tup,rots_tup = get_section_nstaff_nrots_staff_rots(section)
    nstaff,_,nrots,staff_tup,_,rots_tup = get_section_info(section)

    for s in range(nstaff):
        for r in range(nrots):
            #cumulative[ALL_STAFF.index(staff_tup[s]),ALL_ROTS.index(rots_tup[r])] += r_cumulative[s,r] # i think this is double counting
            cumulative[ALL_STAFF.index(staff_tup[s]),ALL_ROTS.index(rots_tup[r])] = r_cumulative[s,r]
            counter[ALL_STAFF.index(staff_tup[s]),ALL_ROTS.index(rots_tup[r])] = r_counter[s,r]
    
def update_rotcounter(collect,v_staff,v_rots,v_rotprod,v_cntr,cuml,cntrs,bias,v_tcost,sect):

	# For reference of matrix dimensions (cumulative, bias, and counter are in the "rotation" context, while curr is in the "shift" context)
	# cumulative = np.zeros((nstaff,nrots),dtype='int64') 
	# counter = np.zeros((nstaff,nrots),dtype='int64')
	# curr = np.zeros((num_staff,num_shifts,num_slots))
 
	num_staff,num_shifts,num_rots,staff_tup,shifts_tup,rots_tup = get_section_info(sect)
	#num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(sect)
	#_,num_rots,_,rots_tup = get_section_nstaff_nrots_staff_rots(sect) 
    
	if sect in CALL_SECTS:
	    num_slots = len(CALL_SLOTS)
	else:
	    num_slots = len(WEEK_SLOTS)

	if collect.SolutionCount() > 0:
		best_solution = collect.SolutionCount() - 1

		curr = np.zeros((num_staff,num_shifts,num_slots))
	    
		for i in range(num_slots):
			for j in range(num_shifts):
				st = collect.Value(best_solution,v_staff[(j,i)])
				if st != -1: # if the rotation is covered by staff (not a placeholder halfday)
					curr[st,j,i] += 1

		for j in range(num_staff):
			for i in range(num_rots):

				# Helpful debugging snippet
				'''new_cnt = collect.Value(best_solution,v_cntr[(j,i)])
				rot_prod = collect.Value(best_solution,v_rotprod[(j,i)])
				rot_val = collect.Value(best_solution,v_rots[(j,i)])
				old_cnt = cntr[j,i]
				print("new count",new_cnt,"=","(old count",old_cnt,"+ bias",bias[j,i],") x rotprod",rot_prod,"where rotation value =",rot_val)'''
				cntrs[1][j,i] = cntrs[0][j,i]
				cntrs[0][j,i] = collect.Value(best_solution,v_cntr[(j,i)])

				cuml[j,i] += collect.Value(best_solution,v_rots[(j,i)])
                #cuml[j,i] = collect.Value(best_solution,v_rots[(j,i)])

		# not updating cumulative b/c it's complex dealing with ndays and not sure if necessary
		return (True, cuml,cntrs,curr,collect.Value(best_solution,v_tcost))
	else:
		print("No solution found for section",sect)
		return (False, sect)

def update_necounter(collect,v_shifts,v_rots,v_rotprod,v_cntr,cuml,cntrs,bias,v_tcost,sect):

    # For reference of matrix dimensions (cumulative, bias, and counter are in the "rotation" context, while curr is in the "shift" context)
    # cumulative = np.zeros((nstaff,nrots),dtype='int64') 
    # counter = np.zeros((nstaff,nrots),dtype='int64')
    # curr = np.zeros((num_staff,num_shifts,num_slots))

    num_slots = len(WEEK_SLOTS)
    
    num_staff,num_shifts,num_rots,staff_tup,shifts_tup,rots_tup = get_section_info(sect)
    #num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(sect)
    #_,num_rots,_,rots_tup = get_section_nstaff_nrots_staff_rots(sect) 

    if collect.SolutionCount() > 0:
        best_solution = collect.SolutionCount() - 1

        curr = np.zeros((num_staff,num_shifts,num_slots))

        for s in range(num_staff):
            for i in range(num_slots):
                shft = collect.Value(best_solution,v_shifts[(s,i)])
                if shft != -1: # if the rotation is covered by staff (not a placeholder halfday)
                    #print("Updating current shift",shifts_tup[shft],"for staff",staff_tup[s],"slot",i)
                    curr[s,shft,i] += 1

        for j in range(num_staff):
            for i in range(num_rots):

                # Helpful debugging snippet
                '''new_cnt = collect.Value(best_solution,v_cntr[(j,i)])
                rot_prod = collect.Value(best_solution,v_rotprod[(j,i)])
                rot_val = collect.Value(best_solution,v_rots[(j,i)])
                old_cnt = cntrs[0][j,i]
                cntrs[1][j,i] = old_cnt
                print("Staff",staff_tup[j],"new count",new_cnt,"=","(old count",old_cnt,"+ bias",bias[j,i],") x rotprod",rot_prod,"where rotation value =",rot_val)'''

                cntrs[1][j,i] = cntrs[0][j,i]
                cntrs[0][j,i] = collect.Value(best_solution,v_cntr[(j,i)])

                cuml[j,i] += collect.Value(best_solution,v_rots[(j,i)])
                #cuml[j,i] = collect.Value(best_solution,v_rots[(j,i)])

        # not updating cumulative b/c it's complex dealing with ndays and not sure if necessary
        return (True, cuml,cntrs,curr,collect.Value(best_solution,v_tcost))
    else:
        print("No solution found for section",sect)
        return (False, sect)

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
        print("                     ===========================================")
        print("                                      WEEK #",int(wk+1))
        print("                     ===========================================")
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

def print_rotcounters(r_cumulative,r_counter,section,tcost=101,bias=None):

    #_,_,staff,rots = get_section_nstaff_nrots_staff_rots(section)  
    _,_,_,staff,_,rots = get_section_info(section)  

    staff_header = '{:>12}'.format('')

    for s in range(len(staff)):
        staff_header += '{:>10}'.format(staff[s])
    staff_header += '{:>8}'.format('Cost')
    print(staff_header)

    for r in range(len(rots)):
        rots_counter = 0
        rots_line = '{:>12}'.format(rots[r])
        for s in range(len(staff)):
            if bias is None:
                b = ''
            else:
                b = bias[s,r]
            rots_line += '{:>3}{:>3}{:>3}|'.format(r_counter[s,r],r_cumulative[s,r],b)
            rots_counter += r_counter[s,r]
        rots_line += '{:>8}'.format(rots_counter)
        print(rots_line)

    print("Total Cost:",tcost)

def print_allcounters(cumulative,counter,section,tcost=101,bias=None):

    if section == 'all':
        staff = ALL_STAFF
        rots = ALL_ROTS
    else:
        #_,_,staff,rots = get_section_nstaff_nrots_staff_rots(section)  
        _,_,_,staff,_,rots = get_section_info(section)  

    staff_header = '{:>12}'.format('')

    for s in range(len(staff)):
        staff_header += '{:>10}'.format(staff[s])
    staff_header += '{:>8}'.format('Cost')
    print(staff_header)

    for r in range(len(rots)):
        rots_counter = 0
        rots_line = '{:>12}'.format(rots[r])
        for s in range(len(staff)):
            if bias is None:
                b = ''
            else:
                b = bias[s,r]
            rots_line += '{:>3}{:>3}{:>3}|'.format(counter[ALL_STAFF.index(staff[s]),ALL_ROTS.index(rots[r])],cumulative[ALL_STAFF.index(staff[s]),ALL_ROTS.index(rots[r])],b)
            rots_counter += counter[ALL_STAFF.index(staff[s]),ALL_ROTS.index(rots[r])]
        rots_line += '{:>8}'.format(rots_counter)
        print(rots_line)

    print("Total Cost:",tcost)

def print_calendar(cal):

    # allows passing either a single week calendar or multiple 
    if len(cal.shape) > 2:
        num_staff, num_slots, num_weeks = cal.shape
        
    #for wk in range(num_weeks):
        for wk in range(num_weeks): # for testing just print the first few weeks
            print()
            print("                     ===========================================")
            print("                                      WEEK #",int(wk+1))
            print("                     ===========================================")
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
                
    else:
        num_staff, num_slots = cal.shape

        print()
        print("                     ===========================================")
        print("                                      SINGLE WEEK")
        print("                     ===========================================")
        print()
        line_header = '{:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}'.format(CALLDAYS[0],CALLDAYS[1],CALLDAYS[2],CALLDAYS[3],CALLDAYS[4],CALLDAYS[5],CALLDAYS[6])
        print(line_header)
        for st in range(num_staff):
            print(ALL_STAFF[st])
            line_am = '{:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}'.format(ALL_SHIFTS[cal[st,0]],ALL_SHIFTS[cal[st,2]],ALL_SHIFTS[cal[st,4]],ALL_SHIFTS[cal[st,6]],ALL_SHIFTS[cal[st,8]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM')]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SUN-AM')]])
            line_pm = '{:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}'.format(ALL_SHIFTS[cal[st,1]],ALL_SHIFTS[cal[st,3]],ALL_SHIFTS[cal[st,5]],ALL_SHIFTS[cal[st,7]],ALL_SHIFTS[cal[st,9]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-PM')]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SUN-PM')]])
            line_call = '{:>25} {:>25} {:>25} {:>25} {:>25}'.format(ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+0]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+1]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+2]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+3]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+4]])
            print(line_am)
            print(line_pm)
            print(line_call)
            
def print_csv_staff_calendar(cal):
    f1=open('./out.csv', 'w+')

    num_staff, num_slots, num_weeks = cal.shape

    #for wk in range(num_weeks):
    for wk in range(num_weeks): # for testing just print the first few weeks
        print()
        print("                     ===========================================")
        print("                                      WEEK #",int(wk+1))
        print("                     ===========================================")
        print()
        line_header = ',{:>0},{:>0},{:>0},{:>0},{:>0},{:>0},{:>0}'.format(CALLDAYS[0],CALLDAYS[1],CALLDAYS[2],CALLDAYS[3],CALLDAYS[4],CALLDAYS[5],CALLDAYS[6])
        print(line_header)
        for st in range(num_staff):
            #print(ALL_STAFF[st])
            line_am = '{:>0},{:>0},{:>0},{:>0},{:>0},{:>0},{:>0},{:>0}'.format(ALL_STAFF[st],ALL_SHIFTS[cal[st,0,wk]],ALL_SHIFTS[cal[st,2,wk]],ALL_SHIFTS[cal[st,4,wk]],ALL_SHIFTS[cal[st,6,wk]],ALL_SHIFTS[cal[st,8,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM'),wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SUN-AM'),wk]])
            line_pm = ',{:>0},{:>0},{:>0},{:>0},{:>0},{:>0},{:>0}'.format(ALL_SHIFTS[cal[st,1,wk]],ALL_SHIFTS[cal[st,3,wk]],ALL_SHIFTS[cal[st,5,wk]],ALL_SHIFTS[cal[st,7,wk]],ALL_SHIFTS[cal[st,9,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-PM'),wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+CALL_SLOTS.index('SUN-PM'),wk]])
            line_call = ',{:>0},{:>0},{:>0},{:>0},{:>0}'.format(ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+0,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+1,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+2,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+3,wk]],ALL_SHIFTS[cal[st,len(WEEK_SLOTS)+4,wk]])
            print(line_am)
            print(line_pm)
            print(line_call)
    f1.close()

'''
=================
 BUILD FUNCTIONS
=================
'''

def build_neshifts(cal,cumulative,r_cumulative,r_counters,section,limit):

    # Settings
    num_staff,num_shifts,num_rots,_,shifts_tup,_ = get_section_info(section)
    #num_staff,num_shifts,_,shifts_tup = get_section_nstaff_nshifts_staff_shifts(section)
    #_,num_rots,_,_ = get_section_nstaff_nrots_staff_rots(section)  
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit
    #print("build_neshifts",shifts_tup)

    # Make a solver with random seed
    solver = make_random_solver()

    # Create constraint variables
    v_neshifts,v_neshifts_flat,v_rots,v_rots_flat,v_cntr,v_cntr_flat,v_rotprod,v_rotprod_flat,v_tcost = create_neshifts(solver,num_slots,section)

    # Constraints
    set_ne_calendar_constraints(solver,v_neshifts,cal,section)

    # Handle holidays
    holidays = get_holidays(cal)

    # Get the bias matrix
    bias = get_bias(section,cumulative)

    # Set the non-essential rotation constraints
    set_ne_constraints(solver,v_neshifts,cal,holidays,section) 

    v_tcost = set_nerotation_constraints(solver,v_neshifts,v_rots,v_cntr,v_rotprod_flat,v_tcost,r_counters,bias,section)

    # Creating decision builder and collector
    collector = get_necollector_obj(solver,v_neshifts_flat,v_rots_flat,v_cntr_flat,v_rotprod_flat,v_tcost,time_limit)

    # test printing the results
    #print_solution(solver,collector,v_staff,v_rots,section)

    counters_result = update_necounter(collector,v_neshifts,v_rots,v_rotprod,v_cntr,r_cumulative,r_counters,bias,v_tcost,section)
    if counters_result[0] is True:
        r_cumulative,r_counter,currwk,tcost = counters_result[1], counters_result[2][0], counters_result[3], counters_result[4] # counters_result[2][1] is the old counter value

        # Use with update counter
        #return (True,cuml,cntrs,currwk)
        return (True,r_cumulative,r_counter,currwk,tcost,bias)
    else:
        return (False, section)

def build_generic_day(cal,cumulative,r_cumulative,r_counters,section,limit):

    # Settings
    num_staff,num_shifts,num_rots,_,_,_ = get_section_info(section)
    #num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts(section)
    #_,num_rots,_,_ = get_section_nstaff_nrots_staff_rots(section)  
    num_slots = len(WEEK_SLOTS)
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create constraint variables
    v_staff,v_staff_flat,v_rots,v_rots_flat,v_cntr,v_cntr_flat,v_rotprod,v_rotprod_flat,v_tcost,v_pcounts = create_variables(solver,num_slots,section)

    # Constraints
    set_day_calendar_constraints(solver,v_staff,cal,section)

    # Handle holidays
    holidays = get_holidays(cal)

    # Get the bias matrix
    bias = get_bias(section,cumulative)

    if section == 'brt':      
        set_brt_constraints(solver,v_staff,cal,holidays)
    elif section == 'sfl':
        set_sfl_constraints(solver,v_staff,cal,holidays)
    elif section == 'msk':
        set_msk_constraints(solver,v_staff,cal,holidays)  
    elif section == 'ner':
        set_ner_constraints(solver,v_staff,cal,holidays)  
    elif section == 'abd':
        set_abd_constraints(solver,v_staff,cal,holidays)  
    elif section == 'cht':
        set_cht_constraints(solver,v_staff,cal,holidays)  
    elif section == 'nuc':
        set_nuc_constraints(solver,v_staff,cal,holidays)  
    elif section == 'sta':
        set_sta_constraints(solver,v_staff,cal,holidays)  
    elif section == 'opr':
        set_opr_constraints(solver,v_staff,cal,holidays)  
    else:
        pass

    v_tcost = set_dayrotation_constraints(solver,v_staff,v_rots,v_cntr,v_rotprod_flat,v_pcounts,v_tcost,r_counters,bias,section,holidays)

    # Creating decision builder and collector
    collector = get_collector_obj(solver,v_staff_flat,v_rots_flat,v_cntr_flat,v_rotprod_flat,v_tcost,time_limit)

    # test printing the results
    #print_solution(solver,collector,v_staff,v_rots,section)

    counters_result = update_rotcounter(collector,v_staff,v_rots,v_rotprod,v_cntr,r_cumulative,r_counters,bias,v_tcost,section)
    if counters_result[0] is True:
        r_cumulative,r_counter,currwk,tcost = counters_result[1], counters_result[2][0], counters_result[3], counters_result[4] # counters_result[2][1] is the old counter value

        # Use with update counter
        #return (True,cuml,cntrs,currwk)
        return (True,r_cumulative,r_counter,currwk,tcost,bias)
    else:
        return (False, section)

def build_generic_call(cal,cumulative,r_cumulative,r_counters,section,limit,regional,week):

	# Settings
	num_staff,num_shifts,num_rots,staff_tup,_,_ = get_section_info(section)
	num_slots = len(CALL_SLOTS)
	time_limit = limit
	
	# Make a solver with random seed
	solver = make_random_solver()

	# Create constraint variables
	v_staff,v_staff_flat,v_rots,v_rots_flat,v_cntr,v_cntr_flat,v_rotprod,v_rotprod_flat,v_tcost,v_pcounts = create_variables(solver,num_slots,section)

	# Constraints
	set_call_calendar_constraints(solver,v_staff,cal,section,regional,week)

	# Handle holidays
	holidays = get_holidays(cal[:,:,week])

	# Get the bias matrix
	bias = get_bias(section,cumulative)

	if section == 'st3':      
		set_st3_constraints(solver,v_staff,cal[:,:,week],holidays,regional,week)
	elif section == 'swg':
		set_swg_constraints(solver,v_staff,cal[:,:,week],holidays)
	elif section == 'stw':
		set_stw_constraints(solver,v_staff,cal[:,:,week],holidays,regional,week)  
	elif section == 'wsp':
		set_wsp_constraints(solver,v_staff,cal[:,:,week],holidays)  
	elif section == 'wmr':
		set_wmr_constraints(solver,v_staff,cal[:,:,week],holidays)  
	elif section == 'nhk':
		set_nhk_constraints(solver,v_staff,cal[:,:,week],holidays,regional,week) 
	else:
		pass
	
	v_tcost = set_callrotation_constraints(solver,v_staff,v_rots,v_cntr,v_rotprod_flat,v_pcounts,v_tcost,r_counters,bias,section,holidays)

	# Creating decision builder and collector
	collector = get_collector_obj(solver,v_staff_flat,v_rots_flat,v_cntr_flat,v_rotprod_flat,v_tcost,time_limit)

	counters_result = update_rotcounter(collector,v_staff,v_rots,v_rotprod,v_cntr,r_cumulative,r_counters,bias,v_tcost,section)

	if counters_result[0] is True:
	    r_cumulative,r_counter,currwk,tcost = counters_result[1], counters_result[2][0], counters_result[3], counters_result[4] # counters_result[2][1] is the old counter value
	    return (True,r_cumulative,r_counter,currwk,tcost,bias)
	else:
		return (False, section)

def build_pooltba_days(calendar,nweeks,limit):
    
    time_limit = limit

    for i in range(nweeks):

        # Make a solver with random seed
        solver = make_random_solver()

        # Create shifts lookup
        v_shifts, v_shifts_flat, v_tcost = create_pooltba_shifts(solver)
    
        # Get the holidays from the calendar
        holidays = get_holidays(calendar[:,:,i])

        # Constraints
	print("Printing pool TBA single week #",i,"BEFORE scheduling")
        print_calendar(calendar[:,:,i])
        v_tcost = set_pooltba_constraints(solver,calendar[:,:,i],v_shifts,v_tcost,holidays)

        # Creating decision builder and collector
        collector = get_collector(solver,v_shifts_flat,v_tcost,time_limit)

        # Print results
        print("updating pooltba calendar")

        #print_pool_results(collector,shifts,pools,nweeks)
        calendar[:,:,i] = update_pooltba_calendar(collector,calendar[:,:,i],v_shifts)
	print("Printing pool TBA single week #",i,"AFTER scheduling")
        print_calendar(calendar[:,:,i])	

    return calendar

def build_pool_days(pools,nweeks,calendar,limit):
    
    time_limit = limit

    # Make a solver with random seed
    solver = make_random_solver()

    # Create shifts lookup
    v_shifts, v_shifts_flat = create_pool_shifts(solver,nweeks,pools)
    
    # Constraints
    set_pool_constraints(solver,pools,nweeks,v_shifts,calendar)

    # Creating decision builder and collector
    collector = get_collector(solver,v_shifts_flat,time_limit)

    # Print results
    #print_pool_results(collector,shifts,pools,nweeks)
    calendar = update_pool_calendar(collector,calendar,v_shifts,nweeks,pools)

    return calendar

def build_multi_day(nweeks,sects,limit,calendar,cumulative,counter):
    
    ndays = len(WEEKDAYS)
    tcost = 0

    for j in range(len(sects)):

        # cumulative and counter are in the "rotation" context
        r_cumulative, r_counters = init_rcounters(cumulative,counter,sects[j])

        for i in range(nweeks):
            print()
            print("===========================================")
            print("          WEEK #",int(i+1)," ",sects[j])
            print("===========================================")

            if sects[j] == 'scv' or sects[j] == 'adm':
                build_result = build_neshifts(calendar[:,:,i],cumulative,r_cumulative,r_counters,sects[j],limit)
            else:
                build_result = build_generic_day(calendar[:,:,i],cumulative,r_cumulative,r_counters,sects[j],limit)      

            if build_result[0] is True:
                r_cumulative,r_counter,recentweek,tcost,bias = build_result[1], build_result[2], build_result[3], build_result[4], build_result[5]
                calendar[:,:,i] = update_day_calendar(recentweek,calendar[:,:,i],sects[j])
                print_rotcounters(r_cumulative,r_counter,sects[j],tcost,bias)
            else:
                return (False,build_result[1])
        update_allcounter(r_cumulative,r_counter,cumulative,counter,sects[j])

    return (True,calendar,tcost)

# newer multi call functions that makes use of minimzation function instead of post-processing analysis
def build_multi_call(nweeks,sects,limit,calendar,cumulative,counter,regional):
    
    for j in range(len(sects)):
       
	# cumulative and counter are in the "rotation" context
        r_cumulative, r_counters = init_rcounters(cumulative,counter,sects[j])

        for wk in range(nweeks):
            print()
            print("===========================================")
            print("          WEEK #",int(wk+1)," ",sects[j])
            print("===========================================")
            

	    #nhklist = get_regional_shifts(sects[j],i,overnights)
	    
	    
	    build_result = build_generic_call(calendar,cumulative,r_cumulative,r_counters,sects[j],limit,regional,wk) # recentweek is to update_calendar matrix

	    if build_result[0] is True:
		    r_cumulative,r_counter,recentweek,tcost,bias = build_result[1], build_result[2], build_result[3], build_result[4], build_result[5]
		    calendar = update_call_calendar(recentweek,calendar,sects[j],wk)
		    print_rotcounters(r_cumulative,r_counter,sects[j],tcost,bias)
	    else:
		    return (False,build_result[1])
        update_allcounter(r_cumulative,r_counter,cumulative,counter,sects[j])

    return (True,calendar,tcost)

def get_regional_shifts(section,week,regional):
	nhk_arr = []
	if (section in NHK_SECTS) and regional:
		for s in range(len(regional)):
			for d in range(1,len(regional[s])):
				if regional[s][d] == i:
					nhk_arr.add(regional[s][0])
	return nhk_arr

def update_day_calendar(cur,cal,sct): # c = nstaff x nhds reflecting 1-week (10 shift); a = calendar matrix
    
    #num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sct)
    num_staff,num_shifts,_,staff,shifts,_ = get_section_info(sct)

    for st in range(num_staff):
        for sh in range(num_shifts):
            for sl in range(len(WEEK_SLOTS)):
                if cur[st,sh,sl] > 0:
                    cal[ALL_STAFF.index(staff[st]),sl] = ALL_SHIFTS.index(shifts[sh])
    return cal

def update_call_calendar(cur,cal,sct,wk): # c = nstaff x nhds reflecting 1-week (10 shift); a = calendar matrix

    #num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sct)
    num_staff,num_shifts,_,staff,shifts,_ = get_section_info(sct)

    for st in range(num_staff):
        for sh in range(num_shifts):
            for sl in range(len(CALL_SLOTS)):
                if cur[st,sh,sl] > 0:
                    cal[ALL_STAFF.index(staff[st]),len(WEEK_SLOTS)+sl,wk] = ALL_SHIFTS.index(shifts[sh])
		    
		    # handle the following week for NH upon scheduling the SUN-PM shift
		    if (sct == 'nhk') and (sl == CALL_SLOTS.index('SUN-PM')) and (cal.shape[2] > wk):
			    for k in range(len(WEEKDAYS)-1): # don't include FRI-PM
				    cal[ALL_STAFF.index(staff[st]),len(WEEK_SLOTS)+k,wk+1] = ALL_SHIFTS.index(shifts[sh])

			    for k in range(CALL_SLOTS.index('FRI-PM'),len(CALL_SLOTS)): # don't include FRI-PM
				    cal[ALL_STAFF.index(staff[st]),len(WEEK_SLOTS)+k,wk+1] = ALL_SHIFTS.index('Day Off')
    return cal 

def update_pool_calendar(collect,cal,shifts,nweeks,pools):
    #staff_calendar = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # staff_calendar matrix is in the "slots" context

    num_solutions = collect.SolutionCount()
    
    for p in range(len(pools)):
        for w in range(nweeks):
            for s in range(len(WEEK_SLOTS)):
                curshift = collect.Value(0,shifts[(p,w,s)]) # just use solution 0 for now
                if curshift != -1:
                    #print("found pool shift",curshift,"for staff",pools[p][0])
                    staffidx = ALL_STAFF.index(pools[p][0])
                    cal[staffidx,s,w] = curshift
    return cal

def update_pooltba_calendar(collect,cal,shifts):
    #staff_calendar = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # staff_calendar matrix is in the "slots" context

    num_solutions = collect.SolutionCount()
    if num_solutions < 1:
        raise ValueError('update_pooltba_calendar: No solutions found for pooltba constraints.')

    for p in range(len(LCM_STAFF)):
        for s in range(len(WEEK_SLOTS)):
            curshift = collect.Value(0,shifts[(p,s)]) # just use solution 0 for now
            if curshift != -1:
                staffidx = ALL_STAFF.index(LCM_STAFF[p])
                cal[staffidx,s] = curshift
    return cal 

def make_random_solver():
    # Creates the solver
    solver = pywrapcp.Solver("Schedule Solution")
    random.seed()
    r = int(random.random()*100000)
    #print("random seed:", r)
    solver.ReSeed(r)

    return solver

def all_staff_idx(s):
    return ALL_STAFF.index(s)

'''
===================
 COUNTER FUNCTIONS
===================
'''

def make_week_hx(cal,cml,cnt,bis):
    num_slots = len(WEEK_SLOTS)+len(CALL_SLOTS)
    
    curr = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64')
    cml = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64')
    cnt = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64')

    for s in range(len(ALL_STAFF)):
        if ALL_STAFF[s] not in LCM_STAFF:
            for slot in range(num_slots):
                shift = cal[s,slot]
                if slot < len(WEEK_SLOTS):
                    if shift == ALL_SHIFTS.index('UCMam Diag 8a-12p') and slot%2 == 0: # the UNC-Diag AM/PM are both considered UNC-Diag
                        curr[s,ALL_ROTS.index('UNC_Diag')] += 1
                    elif shift == ALL_SHIFTS.index('UCMam Proc 8a-12p') and slot%2 == 0:  # the UNC-Proc AM/PM are both considered UNC-Proc
                        if slot == WEEK_SLOTS.index('MON-AM'):
                            curr[s,ALL_ROTS.index('TB')] += 1
                        curr[s,ALL_ROTS.index('UNC_Proc')] += 1
                    elif shift == ALL_SHIFTS.index('FreMam halfday'): # FRE_Mamm
                        curr[s,ALL_ROTS.index('FRE_Mamm')] += 1
                    elif shift == ALL_SHIFTS.index('SL Mam 8a-12p'): # SLN Mamm
                        curr[s,ALL_ROTS.index('SLN_Mamm')] += 1
                    
                    elif (shift == ALL_SHIFTS.index('Fre US/Fluoro 8a-4p') or shift == ALL_SHIFTS.index('SL US/Fluoro 8a-4p')) and slot%2 == 0: # only listed once on qgenda so don't need to sort by slot
                        #curr[s,ALL_ROTS.index('SLN_Sonoflu')] += 1
                        #curr[s,ALL_ROTS.index('FRE_Sonoflu')] += 1
                        if slot == WEEK_SLOTS.index('WED-AM'):
                            curr[s,ALL_ROTS.index('Sonoflu_ICU')] += 1
                        curr[s,ALL_ROTS.index('Sonoflu')] += 1

                    elif shift == ALL_SHIFTS.index('MSK 8a-12p') and slot%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr[s,ALL_ROTS.index('MSK')] += 1
                    
                    elif shift == ALL_SHIFTS.index('Neuro 8a-12p') and slot%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr[s,ALL_ROTS.index('Neuro')] += 1
                        
                    elif shift == ALL_SHIFTS.index('Abdomen 8a-12p') and slot%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        if slot == WEEK_SLOTS.index('MON-AM'):
                            curr[s,ALL_ROTS.index('Abdomen_MON')] += 1
                        curr[s,ALL_ROTS.index('Abdomen')] += 1
                        
                    elif shift == ALL_SHIFTS.index('Chest/PET 8a-12p') and slot%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr[s,ALL_ROTS.index('Chest/PET')] += 1
                        
                    elif shift == ALL_SHIFTS.index('Nucs 8a-4p') and slot%2 == 1: # nucs is a PM rotation only
                        curr[s,ALL_ROTS.index('Nucs')] += 1
                        
                    elif shift == ALL_SHIFTS.index('STAT1 8a-12p') and slot%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr[s,ALL_ROTS.index('STAT_AM')] += 1
                    elif (shift == ALL_SHIFTS.index('STAT1b 12p-4p') or shift == ALL_SHIFTS.index('STAT2 12p-4p')) and slot%2 == 1:
                        curr[s,ALL_ROTS.index('STAT_PM')] += 1
                        
                    elif (shift == ALL_SHIFTS.index('OPPR1am') or shift == ALL_SHIFTS.index('OPPR2am')) and slot%2 == 0: # the AM shifts are indexes 0,1 and the PM shifts are indexes 2,3
                        curr[s,ALL_ROTS.index('OPPR_AM')] += 1
                    elif (shift == ALL_SHIFTS.index('OPPR3pm') or shift == ALL_SHIFTS.index('OPPR4pm')) and slot%2 == 1: 
                        curr[s,ALL_ROTS.index('OPPR_PM')] += 1
                        
                    elif ALL_SHIFTS[shift] in SCV_SHIFTS: # any SCV rotation whether AM/PM counts as one rotation
                        curr[s,ALL_ROTS.index('SCV')] += 1

                    elif shift == ALL_SHIFTS.index('Admin Day') and slot%2 == 0: # counting by half days of Admin time
                        curr[s,ALL_ROTS.index('Admin')] += 2
                    elif shift == ALL_SHIFTS.index('Admin AM') or shift == ALL_SHIFTS.index('Admin PM'):
                        curr[s,ALL_ROTS.index('Admin')] += 1
                else:
                        
                    if shift == ALL_SHIFTS.index('STAT3 4p-11p'):
                        curr[s,ALL_ROTS.index('STAT3')] += 1
                        
                    elif shift == ALL_SHIFTS.index('Swing'):
                        curr[s,ALL_ROTS.index('Swing')] += 1
                        
                    elif shift == ALL_SHIFTS.index('STATWAM 8a-330p'):
                        curr[s,ALL_ROTS.index('STATW')] += 1
                    elif shift == ALL_SHIFTS.index('STATWPM 330p-11p'):
                        curr[s,ALL_ROTS.index('STATW')] += 1
                        
                    elif shift == ALL_SHIFTS.index('WUSPR') and slot%2 == 0:
                        curr[s,ALL_ROTS.index('WUSPR')] += 1
                        
                    elif shift == ALL_SHIFTS.index('WMR') and slot%2 == 0:
                        curr[s,ALL_ROTS.index('WMR')] += 1
                        
    cml += curr
    cnt = add_counter_matrix(cnt+bis,curr)

    return cml,cnt

def schedule_loop(cal_schedule,cumulative,counter,alimit,sections,num_weeks,time_limit,regional=[]):
	
	attempt = 1
        cal_result = (False,False,0) # initialize for the first pass
	
        while cal_result[0] is False and attempt < alimit:
		# preserve the original calendar and counts in case need to revert back due to failed scheduling attempt
		cal_copy = np.copy(cal_schedule) # reset the calendar from the beginning otherwise the old assignments will convert to constraints
                cum_copy = np.copy(cumulative)
                cnt_copy = np.copy(counter)
		
		print("** ATTEMPT",attempt,"** :",sections)

		if sections[0] in CALL_SECTS: # check if we are dealing with call scheduling or day shift scheduling
			cal_result = build_multi_call(num_weeks,sections,time_limit,cal_copy,cum_copy,cnt_copy,regional)
		else:
			cal_result = build_multi_day(num_weeks,sections,time_limit,cal_copy,cum_copy,cnt_copy)

		if cal_result[0] is False:
			section_fail = cal_result[1]
			sections = juggle_sections(sections,section_fail)
			attempt += 1
	
	if attempt < alimit:
		return cal_result[1],cum_copy,cnt_copy,cal_result[2]
	else:
		raise ValueError("No solution could be found after",alimit,"attempts at reshuffling the sections.")

'''
===============
 IMPORT/EXPORT
===============
'''

def import_calendar(fname,cal):
	line = 0
	num_staff = len(ALL_STAFF)
	
	with open(fname, 'rU') as importFile:
		reader = csv.reader(importFile)
		for row in reader:			
			slots = np.array(row).astype(int)
			week = int(line/num_staff)
			staff = int(line%num_staff)
			cal[staff,:,week] = slots
			line += 1
	importFile.close()
			
def export_calendar(cal):
	exportFile=open('./exportCal.csv', 'w+')

	num_staff, num_slots, num_weeks = cal.shape

	for week in range(num_weeks): 
		for staff in range(num_staff):
			line = ''
			for slot in range(num_slots):
				if slot > 0:
					line += ',{:>0}'.format(cal[staff,slot,week])
				else:
					line += '{:>0}'.format(cal[staff,slot,week])
			print(line,file=exportFile)
	exportFile.close()

'''
======
 MAIN
======
'''

def main():

    # Top level parameters
    num_weeks = 4
    tba = False # used to fulfill commitments made to pool rads

    # Solution limits
    alimit = 15
    time_limit = 0 # set to "0" for no limit
    ptime_limit = 0

    # Sections to be schedule
    day_sections = ['brt','cht','nuc','sfl','scv','msk','abd','ner','sta','adm','opr']
    #day_sections = []
    #call_sections = ['st3','swg','stw','wsp','wmr']
    call_sections = ['nhk','stw','st3','swg','wsp','wmr']
    #call_sections = ['st3']

    # Specify import files: f_history = historical schedule; f_schedule = schedule to be filled out; i_schedule = imported schedule
    #f_history = '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/JunAug2018.csv' # history input data
    f_history = False
    f_schedule = False
    i_schedule = False
    #f_schedule = '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/Holiday.csv' # history input data
    #i_schedule = '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/CP_SOLVE/exportCal.csv' # import calendar data

    # Matrix initializations
    cal_init = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # staff_calendar matrix is in the "slots" context
    cal_scheduled = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # staff_calendar matrix is in the "slots" context
    cal_history = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # staff_calendar matrix is in the "slots" context
    cum_init = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64') 
    cnt_init = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64')
    cum_scheduled = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64') 
    cnt_scheduled = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64')
    tcost = 0
    
    # Get historical scheduling data to set counters and cumulative matrix
    if f_history:
	    dept = qa.load_data(f_history)
	    cal_history = qa.qgimport(dept).astype('int64')
            #print_calendar(cal_history)
	    cum_init,cnt_init,bias = init_counter_history(cal_history,cum_init,cnt_init)
	    print_allcounters(cum_init,cnt_init,'all',0,bias)
            
    # Load incomplete department schedule as a starting point (may include vacations, call schedules, time off)
    if f_schedule:
	    dept = qa.load_data(f_schedule)
	    cal_init = qa.qgimport(dept).astype('int64')
    else:
	    holidays = [(0,0)]
	    set_holidays(cal_init,holidays)

    # Set schedules by certain days to work (such as for pools)
    #pooldays = [('CCM',((0,0),(0,2),(0,4)))]
    #pooldays = [('CCM',((0,0),(0,2),(0,4))),
                #('JK',((0,1),(0,3)))]
    #cal_schedule = build_pooltba_days(cal_schedule,num_weeks,ptime_limit)

    # Set cal_schedule constraints
    '''for i in range(num_weeks):
	    set_staffday(cal_init,'RV',i,4,'Day Off')
	    set_staffday(cal_init,'RV',i,2,'Day Off')'''

    # Overnight calls. Numbers listed after nighhawk represent the Friday of the week starting the rotation, first week is week "0".
    # The weekday portions of nightshift carry over into the following week (w+1), though only the starting week is specified.
    # Regional stroke alert calls specify the rotation followed by the week and CALL_SLOT.
    regional_shifts = [('Nightshift 11p-12a',2),
			('NeuroNH 11p-12a',3),
			('Regional Stroke Alert 4p-12a',(1,'SUN-PM'),(2,'SAT-PM'),(2,'THU-PM'),(4,'MON-PM')),
			('Regional Stroke Alert 8a-4p',(2,'SAT-AM'),(3,'SUN-AM'))]

    # Build multiphase call schedule
    if call_sections:
	    cal_scheduled,cum_scheduled,cnt_scheduled,tcost = schedule_loop(cal_init,cum_init,cnt_init,alimit,call_sections,num_weeks,time_limit,regional_shifts)

    # Build multiphase weekday schedule
    if day_sections:
	    if i_schedule:
		    print('Using imported calendar.')
		    import_calendar(i_schedule,cal_scheduled)
		    cum_scheduled,cnt_scheduled,bias = init_counter_history(cal_scheduled,cum_init,cnt_init)
		    print_calendar(cal_scheduled)
	    else:
		    print('Generating calendar for export.')
		    cal_scheduled,cum_scheduled,cnt_scheduled,tcost = schedule_loop(cal_init,cum_init,cnt_init,alimit,day_sections,num_weeks,time_limit)
		    export_calendar(cal_scheduled)
            
	    # if we are handling the TBA pool calendar, start over with the minimum pool use; will need to reset the counters as well
	    if tba:
		    # used in case there is a failure to schedule and need to reset
		    cal_scheduledtba = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # staff_calendar matrix is in the "slots" context
		    
		    set_pooltba_cal(cal_init,cal_scheduled,num_weeks)
		    cal_scheduledtba = build_pooltba_days(cal_init,num_weeks,ptime_limit)
		    print_calendar(cal_scheduledtba)
		    
		    cal_scheduled,cum_scheduled,cnt_scheduled,tcost = schedule_loop(cal_scheduledtba,cum_init,cnt_init,alimit,day_sections,num_weeks,time_limit)
                        
    print_allcounters(cum_scheduled,cnt_scheduled,'all',tcost)
    print_calendar(cal_scheduled)
    print("Time limit:",time_limit)

    #print_csv_staff_calendar(cal_scheduled)
    #print_shift_calendar(shift_calendar)        

if __name__ == "__main__":
  main()
