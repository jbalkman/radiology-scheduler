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
#ALL_SHIFTS = ['----','UNC_Diag_AM','UNC_Diag_PM','UNC_Proc_AM','UNC_Proc_PM','FRE_Mamm','SLN_Mamm','FRE_Sonoflu_AM','FRE_Sonoflu_PM','SLN_Sonoflu_AM','SLN_Sonoflu_PM','MSK_AM','MSK_PM','Neuro_AM','Neuro_PM','Abdomen_AM','Abdomen_PM','Chest/PET_AM','Chest/PET_PM','Nucs','STAT1_AM','STAT1b_PM','STAT2_PM','OPPR1_AM','OPPR2_AM','OPPR3_PM','OPPR4_PM','STAT3','Swing','STATW_AM','STATW_PM','WUSPR','WMR','SCV1_AM','SCV2_AM','SCV3_AM','SCV1_PM','SCV2_PM','Admin','Vaca','Leave'] # make sure all > 'Admin' are forms of leave

ALL_SHIFTS = ("","Vacation",
             "Admin Day",
             "Admin AM",
             "Admin PM",
             "Leave AM",
             "Leave PM",
             "Leave Day",
             "Flex Vaca Day",
             "Day Off",
             "Off AM",
             "Off PM",
             "Meeting",
             "STAT1 8a-12p",
             "STAT1b 12p-4p",
             "STAT2 12p-4p",
             "STAT3 4p-11p",
             "FluoroCall 8a-1159p",
             "Swing",
             "STATWAM 8a-330p",
             "ER PR 1",
             "STATWPM 330p-11p",
             "ER PR 2",
             "Nightshift 11p-12a",
             "Nightshift 1201a-8a",
             "NeuroNH 11p-12a",
             "NeuroNH 1201a-8a",
             "Overnight Fluoro/MRI(a) 11p-12a",
             "Overnight Fluoro/MRI(b) 1201a-8a",
             "Backup Fluoro Call(a) 11p-12a",
             "Backup Fluoro Call(b) 1201a-8a",
             "WMR",
             "WUSPR",
             "OPPR1am",
             "OPPR2am",
             "OPPR3pm",
             "OPPR4pm",
             "TBA",
             "Sonoflu Backup",
             "SCVam abd-chest",
             "SCVam msk-neuro",
             "SCV AM",
             "SCV PM",
             "Abdomen 8a-12p",
             "Abdomen 12-4p",
             "Chest/PET 8a-12p",
             "Chest/PET 12-4p",
             "MSK 8a-12p",
             "MSK 12-4p",
             "Neuro 8a-12p",
             "Neuro 12-4p",
             "UCMam Diag 8a-12p",
             "UCMam Diag 12-4p",
             "UCMam Proc 8a-12p",
             "UCMam Proc 12-4p",
             "FreMam halfday",
             "SL Mam 8a-12p",
             "UCMammoBackup",
             "Nucs 8a-4p",
             "SL US/Fluoro 8a-4p",
             "Fre US/Fluoro 8a-4p",
             "IR:FRE 8a-630p",
             "IR1:SLN 8a-630p",
             "IR3:SLN 8a-630p",
             "IR2:SLN 8a-630p",
             "IR:Admin",
             "IR:Leave",
             "IR:On-Call",
             "Tumor Board",
             "ICU Rds",
             "Holiday",
             "No Call",
             "RSO",
             "Backfill",
             "DFD",
             "Not Working",
             "Manager On-Call",
             "1-844-230-9729")

BRT_SHIFTS = ('UCMam Diag 8a-12p','UCMam Diag 12-4p','UCMam Proc 8a-12p','UCMam Proc 12-4p','FreMam halfday','SL Mam 8a-12p')
SFL_SHIFTS = ('SL US/Fluoro 8a-4p','Fre US/Fluoro 8a-4p') # used to be AM/PM, now switched to be single day
MSK_SHIFTS = ('MSK 8a-12p','MSK 12-4p')
NER_SHIFTS = ('Neuro 8a-12p','Neuro 12-4p') 
ABD_SHIFTS = ('Abdomen 8a-12p','Abdomen 12-4p')
CHT_SHIFTS = ('Chest/PET 8a-12p','Chest/PET 12-4p')
NUC_SHIFTS = ('Nucs 8a-4p')
STA_SHIFTS = ('STAT1 8a-12p','STAT1b 12p-4p','STAT2 12p-4p')
OPR_SHIFTS = ('OPPR1am','OPPR2am','OPPR3pm','OPPR4pm')
ST3_SHIFTS = ('STAT3')
SWG_SHIFTS = ('Swing')
STW_SHIFTS = ('STATWAM 8a-330p','STATWPM 330p-11p')
WSP_SHIFTS = ('WUSPR')
WMR_SHIFTS = ('WMR')
SCV_SHIFTS = ('SCV1_AM','SCV2_AM','SCV3_AM','SCV1_PM','SCV2_PM')

# Rotations - to measure equality
BRT_ROTS = ('UNC_Diag','UNC_Proc','FRE_Mamm','SLN_Mamm')
SFL_ROTS = ('FRE_Sonoflu','SLN_Sonoflu')
MSK_ROTS = ('MSK')
NER_ROTS = ('Neuro')
ABD_ROTS = ('Abdomen')
CHT_ROTS = ('Chest/PET')
NUC_ROTS = ('Nucs')
STA_ROTS = ('STAT_AM','STAT_PM')
OPR_ROTS = ('OPPR_AM','OPPR_PM')
ST3_ROTS = ('STAT3')
SWG_ROTS = ('Swing')
STW_ROTS = ('STATW_AM','STATW_PM')
WSP_ROTS = ('WUSPR')
WMR_ROTS = ('WMR')
SCV_ROTS = ('SCV')

# Staff Lists
ALL_STAFF = ('JDB','SDE','GHL','DCN','JKS','CCM','GJS','GSR','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH','HG','RV','JK','BJK','ATR')
#ALL_STAFF = ('JDB','SDE','GHL','DCN','JKS','CCM','SMN') # used for testing
BRT_STAFF = ('JDB','SDE','GHL','DCN','JKS')
SFL_STAFF = ALL_STAFF
MSK_STAFF = ('CCM','GJS','GSR','DRL','SJP','JK')
NER_STAFF = ('EEP','GSR','JFK','SMN','SJP','BJK','ATR')
ABD_STAFF = ('BCL','DSL','HSS','JKL','SH')
CHT_STAFF = ('BCL','GJS','SMN','RV','JKL')
NUC_STAFF = ('SMN','GSR','HG')
STA_STAFF = ('JDB','SDE','GHL','DCN','JKS','CCM','GJS','GSR','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH','JK','BJK','ATR')
OPR_STAFF = ALL_STAFF
ST3_STAFF = ('JDB','SDE','GHL','DCN','JKS','GJS','GSR','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH','RV')
#ST3_STAFF = ALL_STAFF # used for testing
SWG_STAFF = ALL_STAFF
STW_STAFF = ST3_STAFF
WSP_STAFF = ('JDB','SDE','GHL','DCN','JKS','BCL','DSL','HSS','JKL','HG','RV')
WMR_STAFF = ('GJS','GSR','DRL','SJP','EEP','JFK','SMN','SH')
SCV_STAFF = ALL_STAFF

# General Use
WEEKDAYS = ('MON','TUE','WED','THU','FRI')
CALLDAYS = ('MON','TUE','WED','THU','FRI','SAT','SUN')
WEEK_SLOTS = ('MON-AM','MON-PM','TUE-AM','TUE-PM','WED-AM','WED-PM','THU-AM','THU-PM','FRI-AM','FRI-PM')
CALL_SLOTS = ('MON-PM','TUE-PM','WED-PM','THU-PM','FRI-PM','SAT-AM','SAT-PM','SUN-AM','SUN-PM')
WKND_SECTS = ('stw','wsp','wmr')

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
    elif sect == 'nuc':
        num_staff = len(NUC_STAFF)
        num_rots = len(NUC_ROTS)
        staff = NUC_STAFF
        rots = NUC_ROTS
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
    elif sect == 'scv':
        num_staff = len(SCV_STAFF)
        num_rots = len(SCV_ROTS)
        staff = SCV_STAFF
        rots = SCV_ROTS
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
    elif sect == 'nuc':
        num_staff = len(NUC_STAFF)
        num_shifts = len(NUC_SHIFTS)
        num_rots = len(NUC_ROTS)
        shifts = NUC_SHIFTS
        rots = NUC_ROTS
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
    elif sect == 'scv':
        num_staff = len(SCV_STAFF)
        num_shifts = len(SCV_SHIFTS)
        num_rots = len(SCV_ROTS)
        shifts = SCV_SHIFTS
        rots = SCV_ROTS
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
    elif sect == 'nuc':
        num_staff = len(NUC_STAFF)
        num_shifts = len(NUC_SHIFTS)
        shifts = NUC_SHIFTS
        staff = NUC_STAFF
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
    elif sect == 'scv':
        num_staff = len(SCV_STAFF)
        num_shifts = len(SCV_SHIFTS)
        shifts = SCV_SHIFTS
        staff = SCV_STAFF
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
