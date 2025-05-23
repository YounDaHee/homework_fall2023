import matplotlib.pyplot as plt 
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np

# 로그 파일 경로 설정 (예: runs/experiment1/...)
log_dir_default = ["data_6/q2_pg_pendulum_default_s1_InvertedPendulum-v4_21-05-2025_13-37-11/events.out.tfevents.1747802231.dahee",
                   "data_6/q2_pg_pendulum_default_s2_InvertedPendulum-v4_21-05-2025_13-39-42/events.out.tfevents.1747802382.dahee",
                   "data_6/q2_pg_pendulum_default_s3_InvertedPendulum-v4_21-05-2025_13-42-11/events.out.tfevents.1747802531.dahee",
                   "data_6/q2_pg_pendulum_default_s4_InvertedPendulum-v4_21-05-2025_13-44-43/events.out.tfevents.1747802683.dahee",
                   "data_6/q2_pg_pendulum_default_s5_InvertedPendulum-v4_21-05-2025_13-47-15/events.out.tfevents.1747802835.dahee"]

log_dir_edit = ["data6_1/q2_pg_pendulum_edit_discount0.98_s1_InvertedPendulum-v4_22-05-2025_00-07-20/events.out.tfevents.1747840040.dahee",
                "data6_1/q2_pg_pendulum_edit_discount0.98_s2_InvertedPendulum-v4_22-05-2025_00-09-53/events.out.tfevents.1747840193.dahee",
                "data6_1/q2_pg_pendulum_edit_discount0.98_s3_InvertedPendulum-v4_22-05-2025_00-12-30/events.out.tfevents.1747840350.dahee",
                "data6_1/q2_pg_pendulum_edit_discount0.98_s4_InvertedPendulum-v4_22-05-2025_00-15-06/events.out.tfevents.1747840506.dahee",
                "data6_1/q2_pg_pendulum_edit_discount0.98_s5_InvertedPendulum-v4_22-05-2025_00-17-41/events.out.tfevents.1747840661.dahee"]

event_default = []
event_edit = []
for itr in range(len(log_dir_default)) :
    event_acc = EventAccumulator(log_dir_default[itr])
    event_acc_edit = EventAccumulator(log_dir_edit[itr])
    event_acc.Reload()
    event_acc_edit.Reload()

    temp = event_acc.Scalars('Eval_AverageReturn')
    default = [s.value for s in temp]
    temp = event_acc_edit.Scalars('Eval_AverageReturn')
    edit = [s.value for s in temp]
    
    event_default.append(default)
    event_edit.append(edit)

event_default = np.array(event_default)
event_edit = np.array(event_edit)

default_avg = np.average(event_default, axis=0)
edit_avg = np.average(event_edit, axis=0)

plt.figure()
plt.plot(default_avg, label = 'default')
plt.plot(edit_avg, label = 'edit')
plt.xlabel('step')
plt.ylabel('AVG Return')
plt.title('Compare two pendulum env')
plt.legend()
plt.grid(True)
plt.savefig('img.jpg')