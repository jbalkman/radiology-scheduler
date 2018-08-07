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
        original[fail_idx-1], original[fail_idx] = original[fail_idx], original[fail_idx-1]
        return original
    else:
        sys.exit("Failed on the first section indicating strict constraint problem.")
        
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
    
    num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(section)

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

def create_variables(solver,num_hdays,section):
    v_staff = {}
    v_rots = {}
    v_cntr = {}
    v_rotprod = {}

    v_tcost = solver.IntVar(-500,500, "v_tcost")

    num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(section)
    _,_,num_rots,_,rots_tup = get_section_nstaff_nshifts_nrots_shifts_rots(section)
    #print("create_vars num rots",num_rots)
    #print("create_vars rots tup",rots_tup)

    # the staff matrix returns staff for a given slot and given shift
    for i in range(num_hdays):
        for j in range(num_shifts):
            v_staff[(j,i)] = solver.IntVar(-1, num_staff - 1, "staff(%i,%i)" % (j, i)) # -1 is an escape where shift to not applicable to time of day

    # represents the matrix to be optimized
    for j in range(num_staff):
        for i in range(num_rots):
            v_rots[(j,i)] = solver.IntVar(-1000,1000, "rots(%i,%i)" % (j, i))
            v_cntr[(j,i)] = solver.IntVar(-1000,1000, "v_cntr(%i,%i)" % (j, i))
            v_rotprod[(j,i)] = solver.IntVar(0,1, "v_rotprod(%i,%i)" % (j, i))

    # flattened versions 
    v_staff_flat = [v_staff[(j, i)] for j in range(num_shifts) for i in range(num_hdays)]
    v_rots_flat = [v_rots[(j, i)] for j in range(num_staff) for i in range(num_rots)]
    v_cntr_flat = [v_cntr[(stf,rot)] for stf in range(num_staff) for rot in range(num_rots)]
    v_rotprod_flat = [v_rotprod[(stf,rot)] for stf in range(num_staff) for rot in range(num_rots)]

    max_val = 100 # just set very high
    #scounts = [solver.IntVar(0, max_val, "scount[%i]" % s) for s in range(num_staff)] # overall staff counts (shift slots during the period)
    #pcounts = [solver.IntVar(0, max_val, "pcount[%i]" % s) for s in range(num_staff)] # overall staff counts (shift slots during the period)
    #pcounts = solver.IntVar(0, max_val, "pcounts")
    
    '''for s in range(num_staff):
        print("curr staff:",s)
        #scounts[s] = solver.Sum([solver.IsEqualCstVar(staff[(i,j)],ALL_STAFF.index(staff_tup[s])) for i in range(num_hdays) for j in range(num_shifts)])
        scounts[s] = solver.Sum([solver.IsEqualCstVar(staff[(i,j)],ALL_STAFF.index(staff_tup[s])) for i in range(1) for j in range(1)])'''
    
    poolidx = [staff_tup.index(i) for i in staff_tup if i in LCM_STAFF]
    v_pcounts = solver.Sum([solver.IsMemberVar(v_staff[(j,i)],poolidx) for i in range(num_hdays) for j in range(num_shifts)])

    #print("poolidx",poolidx)
    
    #staffidx = [solver.IntVar(0, len(ALL_STAFF), "staffidx(%i)" % (i)) for i in range(num_staff)]
    #pcounts = solver.Sum([solver.ScalProd(scounts[s],solver.IsMemberVar(ALL_STAFF.index(staff_tup[s]),LCM_STAFF)) for s in range(num_staff)]) # overall pool counts

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

'''
================
 BIAS FUNCTIONS
================
'''

def init_counter_history(cal,cuml,cntr):
    print("initializing counter history...")

    bias = init_general_bias()
    nweeks = cal.shape[2]

    for wk in range(nweeks):
        cuml,cntr = make_week_hx(cal[:,:,wk],cuml,cntr,bias)
    return cuml,cntr,bias

def set_bias_offset(bias,cumulative,section):
    nstaff,nrots,staff,rots = get_section_nstaff_nrots_staff_rots(section)            
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
    else:
        bias = init_adm_bias()

    # Use the cumulative matrix to offset the bias
    if cumulative is None:
        pass
    else:
        set_bias_offset(bias,cumulative,section)
        
    return bias

def init_general_bias():
    bias = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64')
    
    sections = ['brt','cht','nuc','sfl','msk','abd','ner','sta','scv','opr']    
    
    for sect in sections:
        nstaff, nrots, staff_tup, rots_tup = get_section_nstaff_nrots_staff_rots(sect)
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
            bias[OPR_STAFF.index(LCM_STAFF[i]),:] = bias[OPR_STAFF.index(LCM_STAFF[i]),:] - 10
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

def init_scv_bias():
    bias = np.zeros((len(SCV_STAFF),len(SCV_ROTS)),dtype='int64') + 1

    for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in SCV_STAFF:
            bias[SCV_STAFF.index(LCM_STAFF[i]),:] = bias[SCV_STAFF.index(LCM_STAFF[i]),:] - 15
    return bias

def init_adm_bias():
    bias = np.zeros((len(SCV_STAFF),len(SCV_ROTS)),dtype='int64')

    '''for i in range(len(LCM_STAFF)):
        if LCM_STAFF[i] in SCV_STAFF:
            bias[SCV_STAFF.index(LCM_STAFF[i]),:] = bias[SCV_STAFF.index(LCM_STAFF[i]),:] - 5'''
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
        
# May be useful for setting certain amount of admin, SCV time for people
#def set_staff_constraints(solver):
#    pass

def set_rotation_constraints(solver,v_staff,v_rots,v_cntr,v_rotprod_flat,v_tcost,cnts,bis,sect):
    nslts = len(WEEK_SLOTS)
    
    nstaff,nshifts,nrots,shifts,rots_tup = get_section_nstaff_nshifts_nrots_shifts_rots(sect)
    _,_,staff_tup,_ = get_section_nstaff_nshifts_staff_shifts(sect)
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
            solver.Add(v_rots[(stf,rots_tup.index('UNC_Diag'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts.index('UCMam Diag 8a-12p')*nslts+i],stf) for i in range(nslts)]))
            solver.Add(v_rots[(stf,rots_tup.index('UNC_Proc'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts.index('UCMam Proc 8a-12p')*nslts+i],stf) for i in range(nslts)]))
            solver.Add(v_rots[(stf,rots_tup.index('FRE_Mamm'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts.index('FreMam halfday')*nslts+i],stf) for i in range(nslts)]))
            solver.Add(v_rots[(stf,rots_tup.index('SLN_Mamm'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts.index('SL Mam 8a-12p')*nslts+i],stf) for i in range(nslts)]))
            solver.Add(v_rots[(stf,rots_tup.index('TB'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts.index('UCMam Proc 8a-12p')*nslts],stf)]))

            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 3)

    elif sect == 'sfl':
        for stf in range(nstaff):
            #slnSflIdx = [solver.IsEqualCstVar(v_staff_flat[shifts.index('SL US/Fluoro 8a-4p')*nslts+i],stf) for i in range(nslts)]
            #freSflIdx = [solver.IsEqualCstVar(v_staff_flat[shifts.index('Fre US/Fluoro 8a-4p')*nslts+i],stf) for i in range(nslts)]
            #solver.Add(v_rots[(stf,rots.index('Sonoflu'))] == solver.Sum(slnSflIdx+freSflIdx))
            solver.Add(v_rots[(stf,rots_tup.index('FRE_Sonoflu'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts.index('Fre US/Fluoro 8a-4p')*nslts+i],stf) for i in range(nslts)]))
            solver.Add(v_rots[(stf,rots_tup.index('SLN_Sonoflu'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts.index('SL US/Fluoro 8a-4p')*nslts+i],stf) for i in range(nslts)]))
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
                solver.Add(v_rots[(stf,rot)] < 3)

    elif sect == 'ner':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Neuro'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts.index('Neuro 8a-12p')*nslts+i],stf) for i in range(nslts)]))
            
            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 3)

    elif sect == 'abd':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('Abdomen'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[shifts.index('Abdomen 8a-12p')*nslts+i],stf) for i in range(nslts)]))
            
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
                solver.Add(v_rots[(stf,rot)] < 4)
 
    elif sect == 'sta':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('STAT_AM'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(nslts)])) # covers STAT1
            solver.Add(v_rots[(stf,rots_tup.index('STAT_PM'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(nslts,len(STA_SHIFTS)*nslts)])) # covers STAT1b and STAT2
            
            # power constraint that limits number of each rotation that staff takes
            for rot in range(nrots):
                solver.Add(v_rots[(stf,rot)] < 3)
   
    elif sect == 'opr':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('OPPR_AM'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(2*nslts)])) # a hack to cover OPPR1am through OPPR2am indices
            solver.Add(v_rots[(stf,rots_tup.index('OPPR_PM'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(2*nslts,len(OPR_SHIFTS)*nslts)])) # a hack to cover OPPR3pm through OPPR4pm indices
            
            # power constraint that limits number of each rotation that staff takes
            #for rot in range(nrots):
            #    solver.Add(v_rots[(stf,rot)] < 3)

    elif sect == 'scv':
        for stf in range(nstaff):
            solver.Add(v_rots[(stf,rots_tup.index('SCV'))] == solver.Sum([solver.IsEqualCstVar(v_staff_flat[i],stf) for i in range(len(SCV_SHIFTS)*nslts)])) # a hack to cover OPPR1am through OPPR2am indices
            
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

    # Unused cost function code    
    #solver.Add(total_cost == solver.Sum([solver.ScalProd(cnt[s,r]+bis[s,r],solver.IsLessOrEqualCstVar(v_rots[(s,r)],0)) for s in range(nstaff) for r in range(nrots)]))
    #solver.Add(total_cost == solver.Sum([solver.ScalProd(1,v_rots_flat[s*nrots+r]) for s in range(nstaff) for r in range(nrots)]))
    #solver.Add(total_cost == solver.ScalProd(v_rots_flat,cnt_flat+bis_flat))

    # Cost function
    for i in range(nrots*nstaff):
        solver.Add(v_rotprod_flat[i] == solver.IsLessOrEqualCstVar(v_rots_flat[i],0))
        scaling_factor = 1
        solver.Add(v_cntr_flat[i] == v_rotprod_flat[i]*(int((cnt_flat[i]+bis_flat[i])/scaling_factor)))
    solver.Add(v_tcost == solver.Sum([v_cntr_flat[i] for i in range(nrots*nstaff)]))
    
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

def set_day_calendar_constraints(solver,stf,cal,sect):
    num_slots = len(WEEK_SLOTS)

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)

    for i in range(num_staff):
        sect_allstaff_idx = ALL_STAFF.index(staff[i])
        for j in range(num_slots):
            # first check if certain staff is already working during the day or on nights
            if cal[sect_allstaff_idx,j] > 0 or any([cal[sect_allstaff_idx,len(WEEK_SLOTS)+j/2] == ALL_SHIFTS.index(EVE_SHIFTS[k]) for k in range(len(EVE_SHIFTS))]):
                # if that certain staff is already assigned a shift within this section, make that a constrainst in the solution
                if ALL_SHIFTS[cal[sect_allstaff_idx,j]] in shifts:
                    #if sect == 'msk':
                    #    print("Staff",ALL_STAFF[ALL_STAFF.index(staff[i])],"covering shift",ALL_SHIFTS[cal[sect_allstaff_idx,j]])
                    solver.Add(stf[(shifts.index(ALL_SHIFTS[cal[sect_allstaff_idx,j]]),j)] == i)     
                # just make them unavailable for any of the possible section shifts
                else:
                    for k in range(num_shifts):
                        solver.Add(stf[(k,j)] != i)


def set_call_calendar_constraints(solver,stf,cal,sect):

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
                        solver.Add(stf[(k,int(j/2))] != i) # index the PM shift rotations
        elif sect == 'swg':
            #print("handling Swing constraints")
            for j in range(len(WEEK_SLOTS)):
                if cal[sect_allstaff_idx,j] > 0 and cal[sect_allstaff_idx,j] < ALL_SHIFTS.index('Meeting'):
                    for k in range(num_shifts):
                        #print("leave Swing constraint:",k,int(j/2))                        
                        solver.Add(stf[(k,int(j/2))] != i) # index the PM shift rotations
        else: # we are dealing with weekend rotation
            for j in range(len(WEEK_SLOTS)+CALL_SLOTS.index('SAT-AM'),num_slots):
                if cal[sect_allstaff_idx,j] > 0:
                    blocked_wknd = True
            if blocked_wknd and sect in WKND_SECTS:
                for j in range(CALL_SLOTS.index('SAT-AM'),len(CALL_SLOTS)):
                    for k in range(num_shifts):
                        print("leave STATW constraint:",k,j,staff[i])                        
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

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(STA_SHIFTS))],-1))
        
    for i in range(len(WEEKDAYS)):

        # Constraints binding AM/PM rotations
        s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2)] == st[(STA_SHIFTS.index('STAT1b 12p-4p'),i*2+1)])

        # These shifts are real and need to be assigned unless it's a holiday
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
        if i < 4:
            s.Add(st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2)] != st[(STA_SHIFTS.index('STAT1 8a-12p'),i*2+2)])

def set_opr_constraints(s,st,cal,holidays): # s = solver
    
    # Handle special cases for HG
    hg_idx = ALL_STAFF.index('HG')

    for i in range(len(WEEK_SLOTS)):

        # No double coverage
        s.Add(s.AllDifferentExcept([st[(j,i)] for j in range(len(OPR_SHIFTS))],-1))

        # If HG works Nucs in the afternoon, put OPPR in the AM
        if cal[hg_idx,i] == 0: # if HG not scheduled for the morning
            if i%2+1 == 1: # AM case
                s.Add(s.Max([st[(k,i)] == hg_idx for k in range(0,2)]) == 1) # the range specificies the AM OPPR
            else: # PM case
                s.Add(s.Max([st[(k,i)] == hg_idx for k in range(2,4)]) == 1) # the range specificies the PM OPPR
        
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

def set_st3_constraints(s,st): # s = solver
    
    # STAT3 person is for the whole week
    for i in range(len(CALL_SLOTS)-5): # subtract the weekend days to get MON-THU (the last statement will be THU == FRI, that's why only 'til THU)
            s.Add(st[(ST3_SHIFTS.index('STAT3'),i)] == st[(ST3_SHIFTS.index('STAT3'),i+1)])
            
    for i in range(len(CALL_SLOTS)):

        if i < CALL_SLOTS.index('SAT-AM'): 
            # These shifts are real and need to be assigned (MON-FRI STAT3); figure out how to handle the holiday
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

   
def set_holiday_constraint(s,st,day):
    pass
             
'''
====================
 ANALYSIS FUNCTIONS
====================
'''

def update_allcounter(r_cumulative,r_counter,cumulative,counter,section):

    nstaff,nrots,staff_tup,rots_tup = get_section_nstaff_nrots_staff_rots(section)

    for s in range(nstaff):
        for r in range(nrots):
            cumulative[ALL_STAFF.index(staff_tup[s]),ALL_ROTS.index(rots_tup[r])] += r_cumulative[s,r]
            counter[ALL_STAFF.index(staff_tup[s]),ALL_ROTS.index(rots_tup[r])] = r_counter[s,r]
    
def update_rotcounter(collect,v_staff,v_rots,v_rotprod,v_cntr,cuml,cntrs,bias,v_tcost,sect):

    # For reference of matrix dimensions (cumulative, bias, and counter are in the "rotation" context, while curr is in the "shift" context)
    # cumulative = np.zeros((nstaff,nrots),dtype='int64') 
    # counter = np.zeros((nstaff,nrots),dtype='int64')
    # curr = np.zeros((num_staff,num_shifts,num_slots))

    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    
    num_staff,num_shifts,staff_tup,shifts_tup = get_section_nstaff_nshifts_staff_shifts(sect)
    _,num_rots,_,rots_tup = get_section_nstaff_nrots_staff_rots(sect) 

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
                print
                cntrs[1][j,i] = cntrs[0][j,i]
                cntrs[0][j,i] = collect.Value(best_solution,v_cntr[(j,i)])
                cuml[j,i] += collect.Value(best_solution,v_rots[(j,i)])

        # not updating cumulative b/c it's complex dealing with ndays and not sure if necessary
        return (True, cuml,cntrs,curr,collect.Value(best_solution,v_tcost))
    else:
        print("No solution found for section",sect)
        return (False, sect)

def create_analysis(collect,stafflookup,cuml,cntr,bias,sect):
    print("creating analysis...")

    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2

    num_staff,num_shifts,staff,shifts = get_section_nstaff_nshifts_staff_shifts(sect)

    analysis = []
    lowestVal = 100 # some large positive number
    bestSol = []
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
            updated_cuml,cntr_plus = make_brt_hx(curr,cuml,cntr,bias)
        elif sect == 'sfl':
            updated_cuml,cntr_plus = make_sfl_hx(curr,cuml,cntr,bias)
        elif sect == 'msk':
            updated_cuml,cntr_plus = make_msk_hx(curr,cuml,cntr,bias)
        elif sect == 'ner':
            updated_cuml,cntr_plus = make_ner_hx(curr,cuml,cntr,bias)
        elif sect == 'abd':
            updated_cuml,cntr_plus = make_abd_hx(curr,cuml,cntr,bias)
        elif sect == 'cht':
            updated_cuml,cntr_plus = make_cht_hx(curr,cuml,cntr,bias)
        elif sect == 'nuc':
            updated_cuml,cntr_plus = make_nuc_hx(curr,cuml,cntr,bias)
        elif sect == 'sta':
            updated_cuml,cntr_plus = make_sta_hx(curr,cuml,cntr,bias)
        elif sect == 'opr':
            updated_cuml,cntr_plus = make_opr_hx(curr,cuml,cntr,bias)
        elif sect == 'scv':
            updated_cuml,cntr_plus = make_scv_hx(curr,cuml,cntr,bias)
        else:
            raise ValueError('Unresolved section in create_analysis function.')

        # sort matrix by certain criteria
        #analysis.append((sol,np.var(cntr_plus),updated_cuml,cntr_plus,curr))
        #analysis.append((sol,np.sum(cntr_plus),updated_cuml,cntr_plus,curr))
        if np.sum(cntr_plus) < lowestVal:
            lowestVal = np.sum(cntr_plus)
            bestSol = (sol,np.sum(cntr_plus),updated_cuml,cntr_plus,curr)

    #print("sorting analysis of length", len(analysis))
    # finding the best choice of the array
    #analysis.sort(key=lambda x:x[1])
    #bs = analysis[analysis.index(max(analysis,key=itemgetter(1)))]
    #bs = analysis[analysis.index(min(analysis,key=itemgetter(1)))]
    
    #print("Bias Matrix =",bestSol[3])    

    #return bs
    return bestSol
    #return analysis

def create_call_analysis(collect,stafflookup,cuml,cntr,bias,sect):
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
            updated_cuml,cntr_plus = make_st3_hx(curr,cuml,cntr,bias)
        elif sect == 'swg':
            updated_cuml,cntr_plus = make_swg_hx(curr,cuml,cntr,bias)
        elif sect == 'stw':
            updated_cuml,cntr_plus = make_stw_hx(curr,cuml,cntr,bias)
        elif sect == 'wsp':
            updated_cuml,cntr_plus = make_wsp_hx(curr,cuml,cntr,bias)
        elif sect == 'wmr':
            updated_cuml,cntr_plus = make_wmr_hx(curr,cuml,cntr,bias)
        else:
            raise ValueError('Unresolved section in create_call_analysis function.')

        # sort matrix by certain criteria
        #analysis.append((sol,np.var(cntr_plus),updated_cuml,cntr_plus,curr))
        analysis.append((sol,np.sum(cntr_plus),updated_cuml,cntr_plus,curr))

    # finding the best fit
    #print("sorting analysis of length", len(analysis))
    print("finding the max of of this many potential solutions", len(analysis))
    #analysis.sort(key=lambda x:x[1],reverse=True)

    print("Analysis matrix max:", max(analysis,key=itemgetter(1))[1])
    print("Analysis matrix min:", min(analysis,key=itemgetter(1))[1])

    return analysis[analysis.index(max(analysis,key=itemgetter(1)))]

    #return analysis

def print_solution(solver,collect,staff,rots,sect):
    num_staff,num_rots,staff_tup,rots_tup = get_section_nstaff_nrots_staff_rots(sect) 

    print("printing solutions...")
    for sol in range(collect.SolutionCount()):
        for stf in range(num_staff):
            for rot in range(num_rots):
                #print("Staff:",staff_tup[stf],"Rotation:",rots_tup[rot],"Value:",collect.Value(sol,rots[(stf,rot)]))
                print("Staff:",staff_tup[stf],"Rotation:",rots_tup[rot],"Value:",collect.Value(sol,rots[(stf,rot)]))

def print_analysis(solver,collect,stafflookup,anal,sect):
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
    print("Time:", solver.WallTime(), "ms")
    print()

def print_call_analysis(solver,collect,stafflookup,anal,sect):
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
    print("Time:", solver.WallTime(), "ms")
    print()

'''
===================
 RESULTS FUNCTIONS
===================
'''


def print_counter(cuml,cntr,curr,section):

    num_staff,num_rots,staff_tup,rots_tup = get_section_nstaff_nrots_staff_rots(section)
    _,num_shifts,_,shifts_tup = get_section_nstaff_nshifts_staff_shifts(section)
    num_slots = len(WEEK_SLOTS)
 
    print("Current Week Summary for Section", section)
    print("============================")
    print()
    for i in range(num_staff):
        print("Staff",staff_tup[i])
        for j in range(num_shifts):
            mon_am = curr[i][j][0]
            tue_am = curr[i][j][2]
            wed_am = curr[i][j][4]
            thu_am = curr[i][j][6]
            fri_am = curr[i][j][8]
            
            mon_pm = curr[i][j][1]
            tue_pm = curr[i][j][3]
            wed_pm = curr[i][j][5]
            thu_pm = curr[i][j][7]
            fri_pm = curr[i][j][9]
            
            alwk_am = mon_am+tue_am+wed_am+thu_am+fri_am
            alwk_pm = mon_pm+tue_pm+wed_pm+thu_pm+fri_pm
                
            print("AM",shifts_tup[j],int(alwk_am)," MON",int(mon_am),"TUE",int(tue_am),"WED",int(wed_am),"THU",int(thu_am),"FRI",int(fri_am))
            print("PM",shifts_tup[j],int(alwk_pm)," MON",int(mon_pm),"TUE",int(tue_pm),"WED",int(wed_pm),"THU",int(thu_pm),"FRI",int(fri_pm))

    print("Summary for Section", section)
    print("============================")
    print()
    for s in range(num_staff):
        print("Staff",staff_tup[s])
        for r in range(num_rots):
            print("Counter for",rots_tup[r],int(cntr[s][r]))
            print("Cumulative for",rots_tup[r],int(cuml[s][r]))

# this fxn won't work b/c rmvd day dimension in cumulative matrix; use print_calendar instead
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

# this fxn won't work b/c rmvd day dimension in cumulative matrix; use print_calendar instead
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

def print_pool_results(collect,shifts,pools,nweeks):

    num_solutions = collect.SolutionCount()
    print("number of solutions:",num_solutions)

    for sol in range(num_solutions):
        print("Solution #",sol)
        for p in range(len(pools)):
            for w in range(nweeks):
                for s in range(len(WEEK_SLOTS)):
                    sh = collect.Value(sol,shifts[(p,w,s)])
                    print("Staff",pools[p][0],"week",w,"AM/PM",s,"shift",sh)

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

    _,_,staff,rots = get_section_nstaff_nrots_staff_rots(section)  

    staff_header = '{:>12}'.format('')

    for s in range(len(staff)):
        staff_header += '{:>12}'.format(staff[s])
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
            rots_line += '{:>4}{:>4}{:>4}'.format(r_counter[s,r],r_cumulative[s,r],b)
            rots_counter += r_counter[s,r]
        rots_line += '{:>8}'.format(rots_counter)
        print(rots_line)

    print("Total Cost:",tcost)

def print_allcounters(cumulative,counter,section,tcost=101,bias=None):

    if section == 'all':
        staff = ALL_STAFF
        rots = ALL_ROTS
    else:
        _,_,staff,rots = get_section_nstaff_nrots_staff_rots(section)  

    staff_header = '{:>12}'.format('')

    for s in range(len(staff)):
        staff_header += '{:>12}'.format(staff[s])
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
            rots_line += '{:>4}{:>4}{:>4}'.format(counter[ALL_STAFF.index(staff[s]),ALL_ROTS.index(rots[r])],cumulative[ALL_STAFF.index(staff[s]),ALL_ROTS.index(rots[r])],b)
            rots_counter += counter[ALL_STAFF.index(staff[s]),ALL_ROTS.index(rots[r])]
        rots_line += '{:>8}'.format(rots_counter)
        print(rots_line)

    print("Total Cost:",tcost)

def print_calendar(cal):
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

def print_csv_staff_calendar(cal):
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

'''
=================
 BUILD FUNCTIONS
=================
'''

def build_generic(cal,cumulative,r_cumulative,r_counters,section,limit):

    # Settings
    num_staff,num_shifts,_,_ = get_section_nstaff_nshifts_staff_shifts(section)
    _,num_rots,_,_ = get_section_nstaff_nrots_staff_rots(section)  
    num_slots = len(WEEK_SLOTS)
    num_days = num_slots/2
    time_limit = limit

    # Rotation specific counters
    '''r_cumulative = np.zeros((num_staff,num_rots),dtype='int64') 
    r_counter = np.zeros((num_staff,num_rots),dtype='int64')
    r_ocounter = np.zeros((num_staff,num_rots),dtype='int64')
    r_counters = (r_counter, r_ocounter)'''

    # Make a solver with random seed
    solver = make_random_solver()

    # Create constraint variables
    v_staff,v_staff_flat,v_rots,v_rots_flat,v_cntr,v_cntr_flat,v_rotprod,v_rotprod_flat,v_tcost,v_pcounts = create_variables(solver,num_slots,section)

    # Constraints
    set_day_calendar_constraints(solver,v_staff,cal,section)

    # Handle holidays
    holidays = get_holidays(cal)
    print("Holidays:",holidays)

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
    elif section == 'scv':
        set_scv_constraints(solver,v_staff,cal,holidays)  
    else:
        pass

    v_tcost = set_rotation_constraints(solver,v_staff,v_rots,v_cntr,v_rotprod_flat,v_tcost,r_counters,bias,section)

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

def build_st3(cal,cuml,cntr,bias,limit):
    
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
    analysis = create_call_analysis(collector,staff,cuml,cntr,bias,'st3')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'st3')

    return analysis[2],analysis[3],analysis[4]

def build_swg(cal,cuml,cntr,bias,limit):
    
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
    analysis = create_call_analysis(collector,staff,cuml,cntr,bias,'swg')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'swg')

    return analysis[2],analysis[3],analysis[4]

def build_stw(cal,cuml,cntr,bias,limit):
    
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
    analysis = create_call_analysis(collector,staff,cuml,cntr,bias,'stw')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'stw')

    return analysis[2],analysis[3],analysis[4]

def build_wsp(cal,cuml,cntr,bias,limit):
    
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
    analysis = create_call_analysis(collector,staff,cuml,cntr,bias,'wsp')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'wsp')

    return analysis[2],analysis[3],analysis[4]

def build_wmr(cal,cuml,cntr,bias,limit):
    
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
    analysis = create_call_analysis(collector,staff,cuml,cntr,bias,'wmr')

    # Print out the top solution with the least variance
    print_call_analysis(solver,collector,staff,analysis,'wmr')

    return analysis[2],analysis[3],analysis[4]

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

    '''v_staff,v_staff_flat,v_rots,v_rots_flat,v_cntr,v_cntr_flat,v_rotprod,v_rotprod_flat,v_tcost,v_pcounts = create_variables(solver,num_slots,'brt')

    # Constraints
    set_day_calendar_constraints(solver,v_staff,cal,'brt')
    set_brt_constraints(solver,v_staff)
    v_tcost = set_rotation_constraints(solver,v_staff,v_rots,v_cntr,v_rotprod_flat,v_tcost,cntr,bias,'brt')

    # Creating decision builder and collector
    collector = get_collector_obj(solver,v_staff_flat,v_rots_flat,v_cntr_flat,v_rotprod_flat,v_tcost,time_limit)

    # test printing the results
    #print_solution(solver,collector,v_staff,v_rots,'brt')

    # analyze and sort results based on schedule variance
    #analysis = create_analysis(collector,staff,cuml,cntr,bias,'brt')
    cuml,cntr,currwk = update_counter(collector,v_staff,v_rots,v_rotprod,v_cntr,cuml,cntr,bias,'brt')

    # Print out the top solution with the least variance
    #print_analysis(solver,collector,staff,analysis,'brt')
    print_counter(cuml,cntr,currwk,'brt')
                   
    # Use with create analysis
    #return analysis[2],analysis[3],analysis[4]
    
    # Use with update counter
    return cuml,cntr,currwk'''

    return calendar

def build_multi_day(nweeks,sects,limit,calendar,cumulative,counter):
    
    ndays = len(WEEKDAYS)
    tcost = 0

    for j in range(len(sects)):

        # cumulative and counter are in the "rotation" context
        nstaff,nrots,_,_ = get_section_nstaff_nrots_staff_rots(sects[j])  
        r_cumulative = np.zeros((nstaff,nrots),dtype='int64') 
        r_counter = np.zeros((nstaff,nrots),dtype='int64')
        r_ocounter = np.zeros((nstaff,nrots),dtype='int64')
        r_counters = (r_counter, r_ocounter)

        for i in range(nweeks):
            print()
            print("===========================================")
            print("          WEEK #",int(i+1)," ",sects[j])
            print("===========================================")

            #build_result = build_generic(calendar[:,:,i],cumulative,counter,bias,sects[j],limit) # recentweek is to update_calendar matrix            
            build_result = build_generic(calendar[:,:,i],cumulative,r_cumulative,r_counters,sects[j],limit) # recentweek is to update_calendar matrix            
            if build_result[0] is True:
                r_cumulative,r_counter,recentweek,tcost,bias = build_result[1], build_result[2], build_result[3], build_result[4], build_result[5]
                calendar[:,:,i] = update_calendar(recentweek,calendar[:,:,i],sects[j])
                print_rotcounters(r_cumulative,r_counter,sects[j],tcost,bias)
            else:
                return (False,build_result[1])
        update_allcounter(r_cumulative,r_counter,cumulative,counter,sects[j])

    return (True,calendar,tcost)

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
            
        # cumulative and counter are in the "rotation" context
        cumulative = np.zeros((nstaff,nrots,ncshs),dtype='int64') 
        counter = np.zeros((nstaff,nrots),dtype='int64')

        for i in range(nweeks):
            print()
            print("===========================================")
            print("          WEEK #",int(i+1)," ",sects[j])
            print("===========================================")
            
            if sects[j] == 'st3':      
                cumulative,counter,recentweek = build_st3(calendar[:,:,i],cumulative,counter,bias,limit) # recentweek is to update_calendar matrix
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'st3')
            elif sects[j] == 'swg':
                cumulative,counter,recentweek = build_swg(calendar[:,:,i],cumulative,counter,bias,limit)
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'swg')
            elif sects[j] == 'stw':
                cumulative,counter,recentweek = build_stw(calendar[:,:,i],cumulative,counter,bias,limit)
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'stw')
            elif sects[j] == 'wsp':
                cumulative,counter,recentweek = build_wsp(calendar[:,:,i],cumulative,counter,bias,limit)
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'wsp')
            elif sects[j] == 'wmr':
                cumulative,counter,recentweek = build_wmr(calendar[:,:,i],cumulative,counter,bias,limit)
                calendar[:,:,i] = update_call_calendar(recentweek,calendar[:,:,i],'wmr')
            else:
                pass
                
            print_call_results(cumulative,sects[j])
            #print_call_results(counter,sects[j])

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
 COUNTER FUNCTIONS
===================
'''

def make_week_hx(cal,cml,cnt,bis):
    num_slots = len(WEEK_SLOTS)
    
    curr = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64')

    for s in range(len(ALL_STAFF)):
        if ALL_STAFF[s] not in LCM_STAFF:
            for slot in range(num_slots):
                shift = cal[s,slot]
                #print("Shift",shift,ALL_SHIFTS[shift])
                if shift == ALL_SHIFTS.index('UCMam Diag 8a-12p') and slot%2 == 0: # the UNC-Diag AM/PM are both considered UNC-Diag
                    curr[s,ALL_ROTS.index('UNC_Diag')] += 1
                elif shift == ALL_SHIFTS.index('UCMam Proc 8a-12p') and slot%2 == 0:  # the UNC-Proc AM/PM are both considered UNC-Proc
                    curr[s,ALL_ROTS.index('UNC_Proc')] += 1
                elif shift == ALL_SHIFTS.index('FreMam halfday'): # FRE_Mamm
                    curr[s,ALL_ROTS.index('FRE_Mamm')] += 1
                elif shift == ALL_SHIFTS.index('SL Mam 8a-12p'): # SLN Mamm
                    curr[s,ALL_ROTS.index('SLN_Mamm')] += 1
                    
                elif shift == ALL_SHIFTS.index('Fre US/Fluoro 8a-4p') and slot%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                    curr[s,ALL_ROTS.index('FRE_Sonoflu')] += 1
                elif shift == ALL_SHIFTS.index('SL US/Fluoro 8a-4p') and slot%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                    curr[s,ALL_ROTS.index('SLN_Sonoflu')] += 1
                    
                elif shift == ALL_SHIFTS.index('MSK 8a-12p') and slot%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                    curr[s,ALL_ROTS.index('MSK')] += 1
                    
                elif shift == ALL_SHIFTS.index('Neuro 8a-12p') and slot%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                    curr[s,ALL_ROTS.index('Neuro')] += 1
                    
                elif shift == ALL_SHIFTS.index('Abdomen 8a-12p') and slot%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
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
                else:
                    pass

    cml += curr
    cnt = add_counter_matrix(cnt+bis,curr)

    return cml,cnt

# Handle call later
'''elif cur[s,j,i] > 0:
curr[s,rots.index('Swing'),i] += 1

elif shift == ALL_SHIFTS.index('STATWAM 8a-330p'):
# The rotation context works with days whereas the CALL_SLOTS split a morning evening (unlike the weekdays). 
# This is a bit-o-hack to convert the CALL_SHIFT array to the CALLDAYS array for the rotation context 
if i < CALL_SLOTS.index('SUN-AM'):
day_idx = CALLDAYS.index('SAT')
else:
day_idx = CALLDAYS.index('SUN')
curr[s,rots.index('STATW_AM'),day_idx] += 1
elif shift == ALL_SHIFTS.index('STATWPM 330p-11p'):
if i < CALL_SLOTS.index('SUN-AM'):
day_idx = CALLDAYS.index('SAT')
else:
day_idx = CALLDAYS.index('SUN')
curr[s,rots.index('STATW_PM'),day_idx] += 1

elif i > CALL_SLOTS.index('FRI-PM'):
curr[s,rots.index('WUSPR'),i] += 1

elif i > CALL_SLOTS.index('FRI-PM'):
curr[s,rots.index('WMR'),i] += 1'''

def make_brt_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)

    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('brt')
    
    curr_rots = np.zeros((nstaff,nrots),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j < 2 and i%2 == 1: # the UNC-Diag AM/PM are both considered UNC-Diag
                        curr_rots[s,rots.index('UNC_Diag')] += 1
                    elif j < 4 and i%2 == 1:  # the UNC-Proc AM/PM are both considered UNC-Proc
                        curr_rots[s,rots.index('UNC_Proc')] += 1
                    elif j == 4: # FRE_Mamm
                        curr_rots[s,rots.index('FRE_Mamm')] += 1
                    elif j == 5: # SLN Mamm
                        curr_rots[s,rots.index('SLN_Mamm')] += 1
                    else:
                        pass

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_sfl_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('sfl')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Fre US/Fluoro 8a-4p') and i%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('FRE_Sonoflu')] += 1
                    elif j == shifts.index('SL US/Fluoro 8a-4p') and i%2 == 0: # the Sonoflu AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('SLN_Sonoflu')] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_sfl_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_msk_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('msk')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('MSK 8a-12p') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('MSK')] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_msk_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_abd_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('abd')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Abdomen 8a-12p') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('Abdomen')] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_abd_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)
    #cntr_plus = np.multiply(np.sum(curr_rots,axis=2),bis)

    return new_cml,cntr_plus

def make_ner_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('ner')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Neuro 8a-12p') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('Neuro')] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_ner_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_cht_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('cht')
    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Chest/PET 8a-12p') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('Chest/PET')] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_cht_hx function.')

    new_cml = cml.astype('int64')+curr_rots.astype('int64')
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_nuc_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('nuc')
    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('Nucs 8a-4p') and i%2 == 1: # nucs is a PM rotation only
                        curr_rots[s,rots.index('Nucs')] += 1
                    else:
                        pass

    new_cml = cml.astype('int64')+curr_rots.astype('int64')
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_sta_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('sta')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j == shifts.index('STAT1 8a-12p') and i%2 == 0: # the AM/PM are both the same so only need to count the AM rotations
                        curr_rots[s,rots.index('STAT_AM')] += 1
                    elif (j == shifts.index('STAT1b 12p-4p') or j == shifts.index('STAT2 12p-4p')) and i%2 == 1:
                        curr_rots[s,rots.index('STAT_PM')] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_sta_hx function.')
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_opr_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('opr')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0:
                    if j < shifts.index('OPPR3pm') and i%2 == 0: # the AM shifts are indexes 0,1 and the PM shifts are indexes 2,3
                        curr_rots[s,rots.index('OPPR_AM')] += 1
                    elif j > shifts.index('OPPR2am') and i%2 == 1: 
                        curr_rots[s,rots.index('OPPR_PM')] += 1
                    else:
                        pass
                        #raise ValueError('Unresolved shift/halfday combination in make_opr_hx function.')
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_scv_hx(cur,cml,cnt,bis):
    nslts = len(WEEK_SLOTS)
    ndays = len(WEEKDAYS)
    
    nstaff,nshifts,nrots,shifts,rots = get_section_nstaff_nshifts_nrots_shifts_rots('scv')

    curr_rots = np.zeros((nstaff,nrots,ndays),dtype='int64')

    for s in range(nstaff):
        for i in range(nslts):
            for j in range(nshifts):
                if cur[s,j,i] > 0: # any SCV rotation whether AM/PM counts as one rotation
                    curr_rots[s,rots.index('SCV')] += 1
                else:
                    pass
                
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_st3_hx(cur,cml,cnt,bis):
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
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_swg_hx(cur,cml,cnt,bis):
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
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_stw_hx(cur,cml,cnt,bis):
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

                    if j == shifts.index('STATWAM 8a-330p'):
                        curr_rots[s,rots.index('STATW_AM'),day_idx] += 1
                    elif j == shifts.index('STATWPM 330p-11p'):
                        curr_rots[s,rots.index('STATW_PM'),day_idx] += 1
                    else:
                        pass
                        
    new_cml = cml.astype('int64')+curr_rots.astype('int64')      
    #cntr_plus = add_counter_matrix(his,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_wsp_hx(cur,cml,cnt,bis):
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
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

def make_wmr_hx(cur,cml,cnt,bis):
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
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    #cntr_plus = add_counter_matrix(cnt,curr_rots)+bis
    cntr_plus = add_counter_matrix(cnt+bis,curr_rots)

    return new_cml,cntr_plus

'''
======
 MAIN
======
'''

def main():

    # Top level settings
    num_weeks = 4
    time_limit = 1000 # set to "0" for no limit
    day_sections = ['brt','cht','nuc','sfl','msk','abd','ner','sta','scv','opr']
    #day_sections = ['brt','cht','nuc','msk','abd','ner','sta','scv','opr']
    #day_sections = ['brt']
    #call_sections = ['st3','swg','stw','wsp','wmr']
    f_history = '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/JulyAug2018.csv' # history input data
    f_schedule = '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/Holiday.csv' # history input data

    # calinit for keeping track of the schedule by staff; overwritten by qa.qgimport
    cal_schedule = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # staff_calendar matrix is in the "slots" context
    cal_history = np.zeros((len(ALL_STAFF),len(WEEK_SLOTS)+len(CALL_SLOTS),num_weeks),dtype='int64') # staff_calendar matrix is in the "slots" context
    cumulative = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64') 
    counter = np.zeros((len(ALL_STAFF),len(ALL_ROTS)),dtype='int64')
    
    # Get the history to set counter and cumulative matrix
    if f_history:
        dept = qa.load_data(f_history)
        cal_history = qa.qgimport(dept).astype('int64')
        cumulative,counter,bias = init_counter_history(cal_history,cumulative,counter)
        print_allcounters(cumulative,counter,'all',0,bias)
            
    # Get the department information from file
    if f_schedule:
        dept = qa.load_data(f_schedule)
        cal_schedule = qa.qgimport(dept).astype('int64')
    else:
        holidays = [(0,0)]
        set_holidays(cal_schedule,holidays)

    # Set schedules by certain days to work (such as for pools)
    #pooldays = [('CCM',((0,0),(0,2),(0,4)))]
    #pooldays = [('CCM',((0,0),(0,2),(0,4))),
                #('JK',((0,1),(0,3)))]
    #cal_schedule = build_pool_days(pooldays,num_weeks,cal_schedule,time_limit)

    # Set cal_schedule constraints
    for i in range(num_weeks):
        #set_staffday(cal_schedule,'GJS',i,4,'Admin Day')
        #set_staffday(cal_schedule,'GJS',i,2,'Admin Day')
        set_staffday(cal_schedule,'RV',i,4,'Day Off')
        set_staffday(cal_schedule,'RV',i,2,'Day Off')
    #set_staffshift(calinit,'EEP',3,0,1,'OPPR4pm')

    # Build multiphase call schedule
    #if call_sections:
    #    cal = build_multi_call(num_weeks,call_sections,time_limit,cal)

    # Build multiphase weekday schedule
    if day_sections:
        attempt = 1
        cal_result = (False,False) # initialize for the first pass

        # used in case there is a failure to schedule and need to reset 
        cal_backup = np.copy(cal_schedule) 
        cum_backup = np.copy(cumulative)
        cnt_backup = np.copy(counter)

        while cal_result[0] is False and attempt < 100:
            print("** ATTEMPT",attempt,"** :",day_sections)
            cal_result = build_multi_day(num_weeks,day_sections,time_limit,cal_schedule,cumulative,counter)
            if cal_result[0] is False:
                section_fail = cal_result[1]
                day_sections = juggle_sections(day_sections,section_fail)
                cal_schedule = np.copy(cal_backup) # reset the calendar from the beginning otherwise the old assignments will convert to constraints
                cumulative = np.copy(cum_backup)
                counter = np.copy(cnt_backup)
                attempt += 1
        if attempt < 100:
            print_allcounters(cumulative,counter,'all',cal_result[2])
            print_calendar(cal_result[1])
            print("Time limit:",time_limit)
        else:
            print("No solution could be found after 100 attempts at reshuffling the sections.")

    #print_csv_staff_calendar(staff_calendar)
    #print_shift_calendar(shift_calendar)        

if __name__ == "__main__":
  main()
