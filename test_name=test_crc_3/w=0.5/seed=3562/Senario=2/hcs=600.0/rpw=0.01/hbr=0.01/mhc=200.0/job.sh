#!/bin/bash
#$ -M kkosaraj@nd.edu
#$ -m abe
#$ -q long
#$ -pe smp 1
#$ -N name=0.5-3562-2-600.0-0.01-0.01-200.0
#$ -o info
module load conda
conda activate sbl3
/afs/crc.nd.edu/user/k/kkosaraj/.conda/envs/sbl3/bin/python /afs/crc.nd.edu/user/k/kkosaraj/sb3-seir2/argparse_run.py --env_id=gym_seir:seir-b-v0 --weight=0.5 --seed=3562 --Senario=2 --health_cost_scale=600.0 --rho_per_week=0.01 --hospital_beds_ratio=0.01 --max_hospital_cost=200.0 --summary_dir=/afs/crc.nd.edu/user/k/kkosaraj/sb3-seir2/test_name=test_crc_3/w=0.5/seed=3562/w=2/hcs=600.0/rpw=0.01/hbr=0.01/mhc=200.0
