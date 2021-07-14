#!/bin/bash -l

#SBATCH --job-name=test-job

#SBATCH --account=thinklab-ckpt

#SBATCH --partition=ckpt

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=16

#SBATCH --mem=100G

#SBATCH --time=48:00:00

#SBATCH --mail-type=ALL

#SBATCH --mail-user=guanxy@uw.edu

#SBATCH --chdir=/gscratch/thinklab/sb3-seir2

echo "success1"

export PATH=/gscratch/thinklab/anaconda3/lib:$PATH
export LD_LIBRARY_PATH=/gscratch/thinklab/anaconda3/lib:$LD_LIBRARY_PATH

seeds=(1259 133 975 1632 1108 2798 2000 2970 637 1633)

typeset -i variable=$(cat 'Bindex.txt')

seeds=${seeds[@]:$variable}

echo $seeds

for test_name in test_yang_1
do
  for env_id in gym_seir:seir-b-v0
  do
    for weight in 0.5
    do
      for Senario in 0
      do
        for health_cost_scale in 581.0
        do
          for rho_per_week in 0.02
          do
            for hospital_beds_ratio in 0.00287
            do
              for max_hospital_cost in 10.0
              do
                for seed in $seeds
                do
					echo $seed
					/gscratch/thinklab/sb3-seir2/my_scripts/batch.sh $test_name $env_id $weight $Senario $health_cost_scale $rho_per_week $hospital_beds_ratio $max_hospital_cost $seed
					variable=$((variable+1))
					echo $variable > 'Bindex.txt' 
                done
              done
            done
          done
        done
      done
    done
  done
done