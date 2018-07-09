# radiology-scheduler
Constraint Programming Solution for Private Practice Radiology Group
- multiphasic scheduling on week-by-week basis (num weeks configurable)
- balances workload based on historical data
- equalizes workload by minimizing schedule variance between staff
- bias towards scheduling persons who have not worked certain rotations after extended periods of time
- random seed determines thousands of potential schedules which are then sorted to minimize variance
- increase time limit on schedule generation to consider more potential schedules (increases run time)
- global lists define subspecialty staff and all rotations
