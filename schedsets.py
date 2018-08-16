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
              "Regional Stroke Alert 8a-4p",
              "Regional Stroke Alert 4p-12a",
              "Overnight Fluoro/MRI(a) 11p-12a",
              "Overnight Fluoro/MRI(b) 1201a-8a",
              "Backup Fl uoro Call(a) 11p-12a",
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
              "SCV2 AM",
              "SCV2 PM",
              "SCV3 AM",
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

AM_SHIFTS = ("STAT1 8a-12p",
             "OPPR1am",
             "OPPR2am",
             "SCV1 AM",
             "SCV2 AM",
             "Abdomen 8a-12p",
             "Chest/PET 8a-12p",
             "MSK 8a-12p",
             "Neuro 8a-12p",
             "UCMam Diag 8a-12p",
             "UCMam Proc 8a-12p",
             "FreMam halfday",
             "TBA")

PM_SHIFTS = ("STAT2 12p-4p",
             "STAT1b 12p-4p",
             "Nucs 8a-4p",
             "OPPR3pm",
             "OPPR4pm",
             "SCV1 PM",
             "SCV2 PM",
             "SCV3 PM",
             "Abdomen 12-4p",
             "Chest/PET 12-4p",
             "MSK 12-4p",
             "Neuro 12-4p",
             "UCMam Diag 12-4p",
             "UCMam Proc 12-4p",
             "FreMam halfday",
             "TBA")

DAY_SHIFTS = ("SL US/Fluoro 8a-4p",
             "Fre US/Fluoro 8a-4p",
             "Abdomen 8a-12p",
             "Chest/PET 8a-12p",
             "MSK 8a-12p",
             "Neuro 8a-12p",
             "STAT1 8a-12p")

BRT_SHIFTS = ('UCMam Diag 8a-12p','UCMam Diag 12-4p','UCMam Proc 8a-12p','UCMam Proc 12-4p','FreMam halfday','SL Mam 8a-12p')
SFL_SHIFTS = ('SL US/Fluoro 8a-4p','Fre US/Fluoro 8a-4p') # used to be AM/PM, now switched to be single day
MSK_SHIFTS = ('MSK 8a-12p','MSK 12-4p')
NER_SHIFTS = ('Neuro 8a-12p','Neuro 12-4p') 
ABD_SHIFTS = ('Abdomen 8a-12p','Abdomen 12-4p')
CHT_SHIFTS = ('Chest/PET 8a-12p','Chest/PET 12-4p')
NUC_SHIFTS = ('Nucs 8a-4p',)
STA_SHIFTS = ('STAT1 8a-12p','STAT1b 12p-4p','STAT2 12p-4p')
OPR_SHIFTS = ('OPPR1am','OPPR2am','OPPR3pm','OPPR4pm')
ST3_SHIFTS = ('STAT3',)
SWG_SHIFTS = ('Swing',)
STW_SHIFTS = ('STATWAM 8a-330p','STATWPM 330p-11p')
WSP_SHIFTS = ('WUSPR',)
WMR_SHIFTS = ('WMR',)
#SCV_SHIFTS = ('SCV AM','SCV2 AM','SCV3 AM','SCV PM','SCV2 PM') # use this one for standard day shift scheduling
SCV_SHIFTS = ('SCV AM','SCV PM') # use this one for "non-essetial" day shift scheduling
EVE_SHIFTS = ('STAT3 4p-11p','Nightshift 11p-12a','Nightshift 1201a-8a','NeuroNH 11p-12a','NeuroNH 1201a-8a')
ADM_SHIFTS = ('Admin Day','Admin AM','Admin PM')

# Rotations - to measure equality
BRT_ROTS = ('UNC_Diag','UNC_Proc','FRE_Mamm','SLN_Mamm','TB')
#SFL_ROTS = ('FRE_Sonoflu','SLN_Sonoflu')
SFL_ROTS = ('Sonoflu','Sonoflu_ICU') # no need to distinguish between FRE and SLN
MSK_ROTS = ('MSK',)
NER_ROTS = ('Neuro',)
ABD_ROTS = ('Abdomen','Abdomen_MON')
CHT_ROTS = ('Chest/PET',)
NUC_ROTS = ('Nucs',)
STA_ROTS = ('STAT_AM','STAT_PM')
OPR_ROTS = ('OPPR_AM','OPPR_PM')
ST3_ROTS = ('STAT3',)
SWG_ROTS = ('Swing',)
STW_ROTS = ('STATW_AM','STATW_PM')
WSP_ROTS = ('WUSPR',)
WMR_ROTS = ('WMR',)
SCV_ROTS = ('SCV',)
ADM_ROTS = ('Admin',)
ALL_ROTS = BRT_ROTS+SFL_ROTS+MSK_ROTS+NER_ROTS+ABD_ROTS+CHT_ROTS+NUC_ROTS+STA_ROTS+OPR_ROTS+ST3_ROTS+SWG_ROTS+STW_ROTS+WSP_ROTS+WMR_ROTS+SCV_ROTS+ADM_ROTS

# Staff Lists
ALL_STAFF = ('JDB','SDE','HG','SH','JFK','BCL','DSL','JKL','DRL','GHL','SMN','DCN','SJP','EEP','GJS','HSS','JKS','GSr','RV','CCM','JK','RCK','SXK','BJK','SXK')
#ALL_STAFF = ('JDB','SDE','GHL','DCN','JKS','CCM','SMN') # used for testing
BRT_STAFF = ('JDB','SDE','GHL','DCN','JKS','CCM')
SFL_STAFF = ('JDB','SDE','HG','SH','JFK','BCL','DSL','JKL','DRL','GHL','SMN','DCN','SJP','EEP','GJS','HSS','JKS','GSr','RV','CCM','JK')
MSK_STAFF = ('GJS','GSr','DRL','SJP','CCM','JK')
MSV_STAFF = ('GJS','GSr','DRL','SJP','CCM')
NER_STAFF = ('EEP','GSr','JFK','SMN','SJP','BJK')
NSV_STAFF = ('EEP','GSr','JFK','SMN','SJP') 
ABD_STAFF = ('BCL','DSL','HSS','JKL','SH','JK')
ASV_STAFF = ('BCL','DSL','HSS','JKL','SH')
CHT_STAFF = ('BCL','GJS','SMN','RV','JKL','RCK')
NUC_STAFF = ('SMN','GSr','HG','RCK')
STA_STAFF = ('JDB','SDE','GHL','DCN','JKS','CCM','GJS','GSr','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH','CCM','JK','BJK')
OPR_STAFF = ALL_STAFF
ST3_STAFF = ('JDB','SDE','GHL','DCN','JKS','GJS','GSr','DRL','SJP','EEP','JFK','SMN','BCL','DSL','HSS','JKL','SH','RV')
#ST3_STAFF = ALL_STAFF # used for testing
SWG_STAFF = ALL_STAFF
STW_STAFF = ST3_STAFF
WSP_STAFF = ('JDB','SDE','GHL','DCN','JKS','BCL','DSL','HSS','JKL','HG','RV')
WMR_STAFF = ('GJS','GSr','DRL','SJP','EEP','JFK','SMN','SH')
SCV_STAFF = ('JDB','SDE','JFK','BCL','DSL','JKL','DRL','GHL','SMN','DCN','SJP','GJS','HSS','JKS','GSr')
LCM_STAFF = ('CCM','SXK','BJK','JK','RCK')
ADM_STAFF = SCV_STAFF

# General Use
WEEKDAYS = ('MON','TUE','WED','THU','FRI')
CALLDAYS = ('MON','TUE','WED','THU','FRI','SAT','SUN')
WEEK_SLOTS = ('MON-AM','MON-PM','TUE-AM','TUE-PM','WED-AM','WED-PM','THU-AM','THU-PM','FRI-AM','FRI-PM')
CALL_SLOTS = ('MON-PM','TUE-PM','WED-PM','THU-PM','FRI-PM','SAT-AM','SAT-PM','SUN-AM','SUN-PM')
WKND_SECTS = ('stw','wsp','wmr')

# For Pools
CCM_SHIFTS = SFL_SHIFTS+MSK_SHIFTS+STA_SHIFTS+OPR_SHIFTS+('TBA',)
JK_SHIFTS = SFL_SHIFTS+MSK_SHIFTS+STA_SHIFTS+OPR_SHIFTS+ABD_SHIFTS+('TBA',)
RCK_SHIFTS = NUC_SHIFTS+OPR_SHIFTS+CHT_SHIFTS+('TBA',)
BJK_SHIFTS = NER_SHIFTS+OPR_SHIFTS+STA_SHIFTS+('TBA',)
SXK_SHIFTS = STA_SHIFTS+OPR_SHIFTS+('TBA',)

'''
===============
 GET FUNCTIONS
===============
'''

def get_staff_shifts(staff):
    shifts = []

    if staff == 'CCM':
        shifts = CCM_SHIFTS
    elif staff == 'JK':
        shifts = JK_SHIFTS
    elif staff == 'BJK':
        shifts = BJK_SHIFTS
    elif staff == 'RCK':
        shifts = RCK_SHIFTS
    elif staff == 'SXK':
        shifts = SXK_SHIFTS
    else:
        pass
    
    return shifts

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
    elif sect == 'adm':
        num_staff = len(ADM_STAFF)
        num_rots = len(ADM_ROTS)
        staff = ADM_STAFF
        rots = ADM_ROTS
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
    elif sect == 'adm':
        num_staff = len(ADM_STAFF)
        num_shifts = len(ADM_SHIFTS)
        num_rots = len(ADM_ROTS)
        shifts = ADM_SHIFTS
        rots = ADM_ROTS
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
    elif sect == 'adm':
        num_staff = len(ADM_STAFF)
        num_shifts = len(ADM_SHIFTS)
        staff = ADM_STAFF
        shifts = ADM_SHIFTS
    else:
        raise ValueError('Unresolved section name in get_section_nstaff_nshifts_staff_shifts function.')
    
    return num_staff,num_shifts,staff,shifts

def get_section_nstaff_nrots_staff_rots(sect):
    num_staff = 0
    num_rots = 0
    staff = []
    rots = []

    if sect == 'brt':
        num_staff = len(BRT_STAFF)
        num_rots = len(BRT_ROTS)
        rots =  BRT_ROTS
        staff = BRT_STAFF
    elif sect == 'sfl':
        num_staff = len(SFL_STAFF)
        num_rots = len(SFL_ROTS)
        rots =  SFL_ROTS
        staff = SFL_STAFF
    elif sect == 'msk':
        num_staff = len(MSK_STAFF)
        num_rots = len(MSK_ROTS)
        rots =  MSK_ROTS
        staff = MSK_STAFF
    elif sect == 'ner':
        num_staff = len(NER_STAFF)
        num_rots = len(NER_ROTS)
        rots =  NER_ROTS
        staff = NER_STAFF
    elif sect == 'abd':
        num_staff = len(ABD_STAFF)
        num_rots = len(ABD_ROTS)
        rots =  ABD_ROTS
        staff = ABD_STAFF
    elif sect == 'cht':
        num_staff = len(CHT_STAFF)
        num_rots = len(CHT_ROTS)
        rots =  CHT_ROTS
        staff = CHT_STAFF
    elif sect == 'nuc':
        num_staff = len(NUC_STAFF)
        num_rots = len(NUC_ROTS)
        rots =  NUC_ROTS
        staff = NUC_STAFF
    elif sect == 'sta':
        num_staff = len(STA_STAFF)
        num_rots = len(STA_ROTS)
        rots =  STA_ROTS
        staff = STA_STAFF
    elif sect == 'opr':
        num_staff = len(OPR_STAFF)
        num_rots = len(OPR_ROTS)
        rots =  OPR_ROTS
        staff = OPR_STAFF
    elif sect == 'st3':
        num_staff = len(ST3_STAFF)
        num_rots = len(ST3_ROTS)
        rots =  ST3_ROTS
        staff = ST3_STAFF
    elif sect == 'swg':
        num_staff = len(SWG_STAFF)
        num_rots = len(SWG_ROTS)
        rots =  SWG_ROTS
        staff = SWG_STAFF
    elif sect == 'stw':
        num_staff = len(STW_STAFF)
        num_rots = len(STW_ROTS)
        rots =  STW_ROTS
        staff = STW_STAFF
    elif sect == 'wsp':
        num_staff = len(WSP_STAFF)
        num_rots = len(WSP_ROTS)
        rots =  WSP_ROTS
        staff = WSP_STAFF
    elif sect == 'wmr':
        num_staff = len(WMR_STAFF)
        num_rots = len(WMR_ROTS)
        rots =  WMR_ROTS
        staff = WMR_STAFF
    elif sect == 'scv':
        num_staff = len(SCV_STAFF)
        num_rots = len(SCV_ROTS)
        rots = SCV_ROTS
        staff = SCV_STAFF
    elif sect == 'adm':
        num_staff = len(ADM_STAFF)
        num_rots = len(ADM_ROTS)
        rots = ADM_ROTS
        staff = ADM_STAFF
    else:
        raise ValueError('Unresolved section name in get_section_nstaff_nshifts_staff_rots function.')
    
    return num_staff,num_rots,staff,rots

def get_collector_obj(solver,v_staff_flat,v_rots_flat,v_cntr_flat,v_rotprod_flat,v_tcost,tlimit):

    # Create the decision builder.
    #print("creating decision builder...")
    db = solver.Phase(v_staff_flat, solver.CHOOSE_RANDOM, solver.ASSIGN_RANDOM_VALUE)

    # Create the solution collector.
    #print("creating collector...")
    solution = solver.Assignment()
    solution.Add(v_staff_flat)
    solution.Add(v_rots_flat)
    solution.Add(v_cntr_flat)
    solution.Add(v_rotprod_flat)
    solution.Add(v_tcost)

    # Objective
    #objective = solver.Minimize(pcounts, 1)
    objective = solver.Minimize(v_tcost, 1)

    # Create collector
    #collector = solver.AllSolutionCollector(solution)
    collector = solver.LastSolutionCollector(solution)

    if tlimit > 0:
        time_limit_ms = solver.TimeLimit(tlimit)
        solver.Solve(db,[time_limit_ms,objective,collector])
    else:        
        solver.Solve(db,[objective,collector])

    num_solutions = collector.SolutionCount()
    #print("number of solutions:",num_solutions)
    '''for sol in range(num_solutions):
        print("Solution",sol,collector.Value(sol,pcounts))'''
 
    return collector

def get_necollector_obj(solver,v_neshifts_flat,v_rots_flat,v_cntr_flat,v_rotprod_flat,v_tcost,tlimit):

    # Create the decision builder.
    #print("creating decision builder...")
    db = solver.Phase(v_neshifts_flat, solver.CHOOSE_RANDOM, solver.ASSIGN_RANDOM_VALUE)

    # Create the solution collector.
    #print("creating collector...")
    solution = solver.Assignment()
    solution.Add(v_neshifts_flat)
    solution.Add(v_rots_flat)
    solution.Add(v_cntr_flat)
    solution.Add(v_rotprod_flat)
    solution.Add(v_tcost)

    # Objective
    #objective = solver.Minimize(pcounts, 1)
    objective = solver.Minimize(v_tcost, 1)

    # Create collector
    #collector = solver.AllSolutionCollector(solution)
    collector = solver.LastSolutionCollector(solution)

    if tlimit > 0:
        time_limit_ms = solver.TimeLimit(tlimit)
        solver.Solve(db,[time_limit_ms,objective,collector])
    else:        
        solver.Solve(db,[objective,collector])

    num_solutions = collector.SolutionCount()
    #print("number of solutions:",num_solutions)
    '''for sol in range(num_solutions):
        print("Solution",sol,collector.Value(sol,pcounts))'''
 
    return collector

def get_collector(solver,flat,tlimit):

    # Create the decision builder.
    print("creating decision builder...")
    db = solver.Phase(flat, solver.CHOOSE_RANDOM, solver.ASSIGN_RANDOM_VALUE)

    # Create the solution collector.
    print("creating collector...")
    solution = solver.Assignment()
    solution.Add(flat)

    # Create collector
    collector = solver.AllSolutionCollector(solution)

    if tlimit > 0:
        time_limit_ms = solver.TimeLimit(tlimit)
        solver.Solve(db,[time_limit_ms, collector])
    else:        
        solver.Solve(db,[collector])

    return collector
