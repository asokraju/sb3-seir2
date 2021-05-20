#!/bin/bash
#$ -M kkosaraj@nd.edu
#$ -m abe
#$ -q long
#$ -pe smp 1
#$ -N name=0.5-3562-0-600.0-0.03-0.005-100.0
#$ -o info
module load conda
conda activate sbl3
/afs/crc.nd.edu/user/k/kkosaraj/.conda/envs/sbl3/bin/python /afs/crc.nd.edu/user/k/kkosaraj/sb3-seir2/argparse_run.py --env_id=gym_seir:seir-b-v0 --weight=0.5 --seed=3562 --Senario=0 --health_cost_scale=600.0 --rho_per_week=0.03 --hospital_beds_ratio=0.005 --max_hospital_cost=100.0 --summary_dir=/afs/crc.nd.edu/user/k/kkosaraj/sb3-seir2/test_name=test_crc_2/w=0.5/seed=3562/w=0/hcs=600.0/rpw=0.03/hbr=0.005/mhc=100.0
