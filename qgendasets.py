# Fixed Data
Months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

Days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

Holidays = ("New Years", "Martin Luther", "Presidents", "Memorial", "Independence", "Labor", "Thanksgiving", "Christmas")

Rotations = ("","Vacation",
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

RotationsPM = ("Vacation",
             "Admin Day",
             "Admin PM",
             "Leave PM",
             "Leave Day",
             "Flex Vaca Day",
             "Day Off",
             "Off PM",
             "STAT1b 12p-4p",
             "STAT2 12p-4p",
             "OPPR3pm",
             "OPPR4pm",
             "SCV PM",
             "Abdomen 12-4p",
             "Chest/PET 12-4p",
             "MSK 12-4p",
             "Neuro 12-4p",
             "UCMam Diag 12-4p",
             "UCMam Proc 12-4p",
             "FreMam halfday",
             "Nucs 8a-4p",
             "SL US/Fluoro 8a-4p",
             "Fre US/Fluoro 8a-4p",
             "Holiday",
             "Not Working")

RotationsAM = ("Vacation",
             "Admin Day",
             "Admin AM",
             "Leave AM",
             "Leave Day",
             "Flex Vaca Day",
             "Day Off",
             "Off AM",
             "STAT1 8a-12p",
             "OPPR1am",
             "OPPR2am",
             "SCVam abd-chest",
             "SCVam msk-neuro",
             "SCV AM",
             "Abdomen 8a-12p",
             "Chest/PET 8a-12p",
             "MSK 8a-12p",
             "Neuro 8a-12p",
             "UCMam Diag 8a-12p",
             "UCMam Proc 8a-12p",
             "FreMam halfday",
             "SL Mam 8a-12p",
             "SL US/Fluoro 8a-4p",
             "Fre US/Fluoro 8a-4p",
             "Holiday",
             "Not Working")

CallRotations = (
             "STAT3 4p-11p",
             "Swing",
             "STATWAM 8a-330p",
             "ER PR 1",
             "STATWPM 330p-11p",
             "ER PR 2",
             "Nightshift 11p-12a",
             "Nightshift 1201a-8a",
             "NeuroNH 11p-12a",
             "NeuroNH 1201a-8a",
             "WMR",
             "WUSPR")

WeekdayRotations = (
             "STAT1 8a-12p",
             "STAT1b 12p-4p",
             "STAT2 12p-4p",
             "STAT3 4p-11p",
             "OPPR1am",
             "OPPR2am",
             "OPPR3pm",
             "OPPR4pm",
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
             "Nucs 8a-4p",
             "SL US/Fluoro 8a-4p",
             "Fre US/Fluoro 8a-4p")

EssentialRotations = (
             "STAT1 8a-12p",
             "STAT1b 12p-4p",
             "STAT2 12p-4p",
             "STAT3 4p-11p",
             "Swing",
             "OPPR1am",
             "OPPR2am",
             "OPPR3pm",
             "OPPR4pm",
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
             "Nucs 8a-4p",
             "SL US/Fluoro 8a-4p",
             "Fre US/Fluoro 8a-4p",
             "STATWAM 8a-330p",
             "ER PR 1",
             "STATWPM 330p-11p",
             "ER PR 2",
             "WMR",
             "WUSPR")

CustomRotations = (
             "STAT1 8a-12p",
             "STAT1b 12p-4p",
             "STAT2 12p-4p",
             "STAT3 4p-11p",
             "Swing",
             "OPPR1am",
             "OPPR2am",
             "OPPR3pm",
             "OPPR4pm",
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
             "Nucs 8a-4p",
             "SL US/Fluoro 8a-4p",
             "Fre US/Fluoro 8a-4p",
             "STATWAM 8a-330p",
             "ER PR 1",
             "STATWPM 330p-11p",
             "ER PR 2",
             "WMR",
             "WUSPR",
             "SCV",
             "Admin")

MammoRotations = (
             "UCMam Diag 8a-12p",
             "UCMam Diag 12-4p",
             "UCMam Proc 8a-12p",
             "UCMam Proc 12-4p",
             "FreMam halfday",
             "SL Mam 8a-12p")

RelevantRotations = (
             "Vacation",
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
             "Nucs 8a-4p",
             "SL US/Fluoro 8a-4p",
             "Fre US/Fluoro 8a-4p",
             "Tumor Board",
             "ICU Rds",
             "Holiday",
             "DFD",
             "Not Working")

CallRotations = (
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
             "WMR",
             "WUSPR")

ATORotations = (
             "Swing",
             "STATWAM 8a-330p",
             "ER PR 1",
             "STATWPM 330p-11p",
             "ER PR 2",
             "Nightshift 11p-12a",
             "Nightshift 1201a-8a",
             "NeuroNH 11p-12a",
             "NeuroNH 1201a-8a",
             "WMR",
             "WUSPR")

WkndCallRotations = (
             "STATWAM 8a-330p",
             "ER PR 1",
             "STATWPM 330p-11p",
             "ER PR 2",
             "WMR",
             "WUSPR")

WorkRotations = (
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
             "Nucs 8a-4p",
             "SL US/Fluoro 8a-4p",
             "Fre US/Fluoro 8a-4p",
             "Tumor Board",
             "ICU Rds")

OffRotations = (
             "Holiday",
             "Vacation",
             "Leave AM",
             "Leave PM",
             "Leave Day",
             "Flex Vaca Day",
             "Day Off",
             "Off AM",
             "Off PM",
             "Not Working")

Units = {    "": 0,
             "Vacation" : 2,
             "Admin Day": 2,
             "Admin AM" : 1,
             "Admin PM" : 1,
             "Leave AM" : 1,
             "Leave PM" : 1,
             "Leave Day" : 2,
             "Flex Vaca Day" : 2,
             "Day Off" : 2,
             "Off AM" : 1,
             "Off PM" : 1,
             "Meeting" : 0,
             "STAT1 8a-12p" : 1,
             "STAT1b 12p-4p" : 1,
             "STAT2 12p-4p" : 1,
             "STAT3 4p-11p" : 2,
             "FluoroCall 8a-1159p" : 0,
             "Swing" : 1.25,
             "STATWAM 8a-330p" : 2,
             "ER PR 1" : 0.75,
             "STATWPM 330p-11p" : 2,
             "ER PR 2" : 0.75,
             "Nightshift 11p-12a" : 0,
             "Nightshift 1201a-8a" : 2.5,
             "NeuroNH 11p-12a" : 0,
             "NeuroNH 1201a-8a" : 2.5,
             "Overnight Fluoro/MRI(a) 11p-12a" : 0,
             "Overnight Fluoro/MRI(b) 1201a-8a" : 0,
             "Backup Fluoro Call(a) 11p-12a" : 0,
             "Backup Fluoro Call(b) 1201a-8a" : 0,
             "WMR" : 1.25,
             "WUSPR" : 1.25,
             "OPPR1am" : 1,
             "OPPR2am" : 1,
             "OPPR3pm" : 1,
             "OPPR4pm" : 1,
             "TBA" : 0,
             "Sonoflu Backup" : 0,
             "SCVam abd-chest" : 1,
             "SCVam msk-neuro" : 1,
             "SCV AM" : 1,
             "SCV PM" : 1,
             "Abdomen 8a-12p" : 1,
             "Abdomen 12-4p" : 1,
             "Chest/PET 8a-12p" : 1,
             "Chest/PET 12-4p" : 1,
             "MSK 8a-12p" : 1,
             "MSK 12-4p" : 1,
             "Neuro 8a-12p" : 1,
             "Neuro 12-4p" : 1,
             "UCMam Diag 8a-12p" : 1,
             "UCMam Diag 12-4p" : 1,
             "UCMam Proc 8a-12p" : 1,
             "UCMam Proc 12-4p" : 1,
             "FreMam halfday" : 1,
             "SL Mam 8a-12p" : 1,
             "UCMammoBackup" : 0,
             "Nucs 8a-4p" : 1,
             "SL US/Fluoro 8a-4p" : 2,
             "Fre US/Fluoro 8a-4p" : 2,
             "IR:FRE 8a-630p" : 2,
             "IR1:SLN 8a-630p" : 2,
             "IR3:SLN 8a-630p" : 2,
             "IR2:SLN 8a-630p" : 2,
             "IR:Admin" : 2,
             "IR:Leave" : 2,
             "IR:On-Call" : 0,
             "Tumor Board" : 0.25,
             "ICU Rds" : 0,
             "Holiday" : 2,
             "No Call" : 0,
             "RSO" : 0,
             "Backfill" : 0,
             "DFD" : 0,
             "Not Working" : 2,
             "Manager On-Call" : 0,
             "1-844-230-9729" : 0
             }

Pools = (    "(LLS)",
             "(RCK)",
             "(CCM)",
             "(ATR)",
             "(SXK)",
             "(BJK)",
             "(JK)",
             "(EMR)",
             "(Oak)")
