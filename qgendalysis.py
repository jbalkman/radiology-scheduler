import csv
import numpy as np

# Globals
OFFSET = 2
START_PRINT = 7
END_PRINT = 7
DATES_LINE = 6-OFFSET
DAYS_LINE = 7-OFFSET

# Classes
class Staff:
    def __init__(self, r, d, s):
        self.name = r[0]
        self.dates = d
        self.startday = s
        self.firsts, = np.where(d == 1)
        self.schedule = []
        self.add_row(r)
        self.tunits = 0
        self.ounits = 0
        self.wkdays = 0
        self.wknds = 0

    def get_wkdays(self):
        return self.wkdays

    def add_row(self, r):
        for i in range(len(r)-1):
            self.schedule.append(Rotations.index(r[i+1]))

    def finalize(self):
         self.schedule = np.reshape(self.schedule, newshape=(len(self.schedule)/len(self.dates), len(self.dates)))
         self.sshape = self.schedule.shape
         self.tunits = self.get_total_u()
         self.ounits = self.get_total_off_u()
         for d in range(len(self.dates)):
             if not self.iswknd(d):
                 self.wkdays += 1
         self.wknds = len(self.dates) - self.wkdays

    def runtests(self):
        for i in range(len(self.dates)):
            print "Index: ", i, ", ", self.iswknd(i)
                
    def p_sched(self):
        print self.name
        for i in range(self.sshape[0]):
            for j in range(self.sshape[1]):
                print Rotations[self.schedule[i,j]],
                if j < self.sshape[1]-1:
                    print '\t\t\t',
                else:
                    print '\n'  
            print '\n'
        print '\n\n'

    def p_allrotation_u(self):
        RotationTuple = Rotations
        tally = 0
        working_units = self.tunits-self.ounits
        day_units = 0

        for i in range(len(WeekdayRotations)):
            day_units += self.get_rotation_u(WeekdayRotations[i], 1, 1, 12, 31)        

        for i in range(len(RotationTuple)):
            units = self.get_rotation_u(RotationTuple[i], 1, 1, 12, 31)
            tally += units
            if (units > 0):
                rotation_str = RotationTuple[i]+" units for "+self.name+": "
                print rotation_str.ljust(45, " ")+str(units).rjust(6, " ")+" ("+str(int(100*units/day_units))+"% of day units, "+str(int(100*units/working_units))+"% of working units, "+str(int(100*units/self.tunits))+"% of total units)"
        total_units_str = "Total units for "+self.name+": "
        print total_units_str.ljust(45, " ")+str(self.tunits)+" total units - "+str(self.ounits)+" off units = "+str(working_units)+" working units"

    def get_total_u(self, mylist = None):
        total_units = 0
        rotation_units = 0

        if mylist == None:
            mylist = Rotations

        for i in range(len(mylist)):
            total_units += self.get_rotation_u(mylist[i], 1, 1, 12, 31)
        return total_units

    def get_total_off_u(self):
        total_units = 0
        for i in range(len(OffRotations)):
            total_units += self.get_rotation_u(OffRotations[i], 1, 1, 12, 31)
        return total_units   

    def p_day(self, m, d):
        dayidx = self.get_dayidx(m, d)
        for i in range(s[0]):
            print Rotations[self.schedule[i,dayidx]]

    def iswknd(self, idx):
        if (((idx+self.startday)%7 == 6) or ((idx+self.startday)%7 == 5)):
            return True
        return False

    def p_wknds(self, m):
        start_idx = self.get_dayidx(m, 1)
        if m == 12:
            end_idx = self.get_dayidx(m,1)+30
        else:
            end_idx = self.get_dayidx(m+1,1)-1

        for i in range(self.sshape[0]):
            for d in range(start_idx, end_idx+1):
                if self.iswknd(d):
                    print Rotations[self.schedule[i,d]]

    def get_dayidx(self, m, d):
        start_idx = self.firsts[m-1]
        day = start_idx + (d-1)
        return day

    def get_name(self):
        return self.name

    # Get number of units for a given rotation with the following date range (sm = start month, sd = start day, em = end month, ed = end day)
    def get_rotation_u(self, r, sm, sd, em, ed):
        units = 0
        start_idx = self.get_dayidx(sm, sd)
        end_idx = self.get_dayidx(em, ed)

        for d in range(start_idx,end_idx+1):
            units += self.get_rotation_day_u(r, d)
        return units

    def p_rotation_u(self, r, sm, sd, em, ed):
        units = self.get_rotation_u(r, sm, sd, em, ed)

        rotation_str = r+" units for "+self.name+": "
        print rotation_str.ljust(45, " ")+str(units).rjust(6, " ")+" units"
        
    def get_rotation_day_u(self, r, d):
        units = 0
  
        for i in range(self.sshape[0]):
            rotation = Rotations[self.schedule[i,d]]
            if r in rotation:
                units += Units[rotation]
        return units

    def p_missing_staff_u(self):
        missing_units = 0
        for d in range(self.sshape[1]):
            if not self.iswknd(d):
                dayunits = 0
                for j in range(self.sshape[0]):
                    rotation = Rotations[self.schedule[j,d]]
                    dayunits += Units[rotation]
                if dayunits < 2:
                    missing_units += 2 - dayunits
                    month, day = self.get_monthday(d)
                    #print str(month)+"/"+str(day)+" units = "+str(missing_units)
        missing_units_str = "Missing units for "+self.name+": "
        print missing_units_str.ljust(45, " ")+str(missing_units) 

        return missing_units

    def get_monthday(self, idx):
        if idx >= self.firsts[-1]:
            month = 12
        else:
            month = np.where(self.firsts > idx)[0][0]
        day = idx-self.firsts[month-1]+1
            
        return month, day

class Dept:
    def __init__(self, f):
        self.fname = f
        self.staff = []

    def add_staff(self, s):
        self.staff.append(s)
    
    def p_staff_day(self, inits, m, d):
        s = self.get_staff(inits)
        if s:
            s.p_day(m, d)
        else:
            print "Error: staff not found."

    def p_staff_wknds(self, inits, m):
        s = self.get_staff(inits)
        if s:
            s.p_wknds(m)
        else:
            print "Error: staff not found."

    def get_staff(self, n):
        for i in range (len(self.staff)):
            if n in self.staff[i].get_name():
                return self.staff[i]
        return None

    def get_staff_names(self):
        staff_names = []
        for i in range (len(self.staff)):
            staff_names.append(self.staff[i].get_name())
        return staff_names

    '''def p_staff_rotation_u(self, n, r, sm, sd, em, ed):
        units = 0
        s = self.get_staff(n)
        units = s.get_rotation_u(r, sm, sd, em, ed)
        units_str = str(units).rjust(10, " ")
        print "\n"+rotation+" units for staff "+n+" between "+str(sm)+"/"+str(sd)+" and "+str(em)+"/"+str(ed)+": "+units_str+"\n"
        print "Source: "+self.fname+"\n"
        print "======================================================================================="

        return units'''

    def p_staff_rotation_monthly_u(self, n, r, sm, em):
        s = self.get_staff(n)

        t_units = 0
        for m in range(sm, em+1):
            m_units = 0
            if m == 12:
                m_units += s.get_rotation_u(r, m, 1, m, 31)
            else:
                m_units += s.get_rotation_u(r, m, 1, m+1, 0)
            print Months[m-1]+": "+str(m_units)
            t_units += m_units
        t_units_str = str(t_units).rjust(10, " ")
        print "\n"+rotation+" units for staff "+n+" between "+Months[sm-1]+" and "+Months[em-1]+": "+t_units_str+"\n"
        print "Source: "+self.fname+"\n"
        print "======================================================================================="

        return t_units

    # Print total dept units for certain activity
    def p_dept_rotation_monthly_u(self, r, sm, em):
        t_units = 0
        for m in range(sm, em+1):
            m_units = 0
            for i in range(len(self.staff)):
                if m == 12:
                    m_units += self.staff[i].get_rotation_u(r, m, 1, m, 31)
                else:
                    m_units += self.staff[i].get_rotation_u(r, m, 1, m+1, 0)
            print Months[m-1]+": "+str(m_units)
            t_units += m_units
        t_units_str = str(t_units).rjust(10, " ")
        print "\n"+r+" units for dept between "+Months[sm-1]+" and "+Months[em-1]+": "+t_units_str+"\n"
        print "Source: "+self.fname+"\n"
        print "======================================================================================="
        return t_units

    def p_dept_rotation_u(self, r, sm, sd, em, ed):
        t_units = 0
        for i in range(len(self.staff)):
            t_units += self.staff[i].get_rotation_u(r, sm, sd, em, ed)
        t_units_str = str(t_units).rjust(10, " ")
        print r+" units for dept between "+str(sm)+"/"+str(sd)+" and "+str(em)+"/"+str(ed)+": "+t_units_str            

    def p_dept_staff_rotation_u(self, r, sm, sd, em, ed):
        t_units = 0
        s_units = 0
        for i in range(len(self.staff)):
            s_units = self.staff[i].get_rotation_u(r, sm, sd, em, ed)
            if (s_units > 0):
                s_units_str = str(t_units).rjust(10, " ")
                staff_str = "Total "+r+" units for staff "+self.staff[i].get_name()+": " 
                print staff_str.ljust(45, " ")+str(s_units)+" units"
                t_units += s_units
        print "Total "+r+" units for dept: "+str(t_units)+" units"

    def p_dept_pool_u(self, sm, em, rotlist=None):
        t_units = 0

        if rotlist == None:
            rotlist = Rotations

        for i in range(len(Pools)):
            p_units = 0
            name = Pools[i]
            s = self.get_staff(name)
            p_units = s.get_total_u(mylist = rotlist)
            t_units += p_units
            if (p_units > 0):
                pool_str = "Total units for pool "+name+": " 
                print pool_str.ljust(30, " ")+str(p_units)+" units"
        print "Total pool units: "+str(t_units)+" units"

    def p_dept_staff_u(self, sm, em, rotlist=None, sl=None):
        t_units = 0
        staff_list = []

        if rotlist == None:
            rotlist = Rotations

        if sl == None:
            staff_list = self.get_staff_names()
        else:
            staff_list = sl

        for i in range(len(staff_list)):
            s = self.get_staff(staff_list[i])
            s_units = 0
            s_units = s.get_total_u(mylist = rotlist)
            t_units += s_units 
            if (s_units > 0):
                staff_str = "Total units for staff "+s.get_name()+": " 
                print staff_str.ljust(30, " ")+str(s_units)+" units"
        print "Total staff units: "+str(t_units)+" units"
    
    def p_staff_missing_u(self, sl=None):
        staff_list = []

        if sl == None:
            staff_list = self.get_staff_names()
        else:
            staff_list = sl

        for i in range(len(staff_list)):
            s = self.get_staff(staff_list[i])
            s.p_missing_staff_u()

    def p_dept_missing_u(self):
        missing_units = 0
        for s in range(len(self.staff)):
            missing_units += self.staff[s].p_missing_staff_u()
                
        print "Missing units for department: "+str(missing_units)
        return missing_units

    def csv_dept_staff_overhead(self, flist, rotlist=None, sl=None):
        staff_list = []
        r_units = 0
        t_units = 0
        comma_str = ""

        if sl == None:
            staff_list = self.get_staff_names()
        else:
            staff_list = sl

        if rotlist == None:
            rotlist = EssentialRotations
            
        # make comma str
        for k in range(len(flist)-1):
            comma_str += ","
            
        # print row headings
        
        for j in range(len(rotlist)):
            if "SCV" in rotlist[j]:
                print ",,"+rotlist[j],
            else:
                print","+rotlist[j]+comma_str,
        print ""
            
        for i in range(len(self.staff)):
            print self.staff[i].get_name()+",",
            t_units = 0
            for j in range(len(rotlist)):
                if "SCV" in rotlist[j]:
                    print str(t_units)+",",
                    t_units = 0
                r_units = self.staff[i].get_rotation_u(rotlist[j], 1, 1, 12, 31)
                t_units += r_units
                print str(r_units)+",",
            print str(t_units)

def get_department_matrix(dept):
    pass

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

# Functions
def load_data(fname):
    start_day = 0
    dates = []
    dept = Dept(fname)
    line = 0
    staff = None
    
    # Store Data
    with open(fname, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if (line == DATES_LINE):
                row.pop(0)
                dates = np.array(row).astype(int)
            elif (line == DAYS_LINE):
                row.pop(0)
                start_day = Days.index(row[0])
            elif (row[0] != '' and line > DAYS_LINE):
                if staff != None:
                    staff.finalize()
                    dept.add_staff(staff)
                staff = Staff(row, dates, start_day)
            elif staff:
                staff.add_row(row)
            line += 1
    return dept

# MAIN
if __name__ == '__main__':
    sm = 1
    sd = 1
    em = 5
    ed = 31
    staff = "(SDE)"
    staff_list = ['(SMN)', '(BCL)']
    rotation = 'Admin'
    rotations = ['Admin', 'SCV']
    files = ['/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/Staff_2014.csv',
             '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/Staff_2015.csv',
             '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/Staff_2016.csv',
             '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/Staff_2017.csv',
             '/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/Staff_2018.csv']

    #files = ['/Users/jasonbalkman/Documents/KAISER/SCHEDULE_ANALYSIS/DATA/Staff_2018.csv']

    for i in range(len(files)):
        fname = files[i]
        print "\n"+"File Analysis: "+fname
        print "=================="
        d = load_data(fname)
        d.p_dept_staff_rotation_u(rotation, sm, sd, em, ed)
        
        d.p_dept_missing_u()
    #d.p_dept_staff_u(sm, em, rotlist=WeekdayRotations)
    #d.p_dept_staff_u(sm, em, rotlist=OffRotations)
    #d.p_staff_missing_u(staff_list)

    #print "ATO ROTATIONS"
    #d.p_dept_staff_u(sm, em, rotlist=ATORotations, sl=staff_list)
    #print "OFF WORK"
    #d.p_dept_staff_u(sm, em, rotlist=OffRotations, sl=staff_list)
    #print "DAY ROTATIONS"
    #d.p_dept_staff_u(sm, em, rotlist=WeekdayRotations, sl=staff_list)
    #print "ADMIN"
    #d.p_dept_staff_u(sm, em, rotlist=['Admin'], sl=staff_list)
    #d.p_staff_rotation_monthly_u(staff, rotation, sm, em)
    #d.csv_dept_staff_overhead(rotlist=CustomRotations)
