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

typeset -i variable=$(cat 'Bindex5.txt')

echo $variable
current=0

for test_name in test_yang_5_test
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
            for hospital_beds_ratio in 0.010 #0.005
            do
              for max_hospital_cost in 10.0 #5.0
              do
                for seed in 2345 #2901
                do
                  for learning_rate in 0.0003 0.0002
                  do
                    for clip_range in 0.10 0.01
                    do
                      for rl_algo in 0
                      do
						current=$((current+1))
						if [ "$current" -gt "$variable" ]
						then
							echo $current
							/gscratch/thinklab/sb3-seir2/my_scripts_Hyak/batch.sh $test_name $env_id $weight $Senario $health_cost_scale $rho_per_week $hospital_beds_ratio $max_hospital_cost $seed $learning_rate $clip_range $rl_algo
							variable=$((variable+1))
							echo $variable > 'Bindex5.txt'
						fi
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done


# #!/bin/bash -l

# for test_name in test_05_28
# do
#   for env_id in gym_seir:seir-b-v0
#   do
#     for weight in 0.5
#     do
#       for Senario in 0 1 2
#       do
#         for health_cost_scale in 600.0
#         do
#           for rho_per_week in 0.02
#           do
#             for hospital_beds_ratio in 0.005
#             do
#               for max_hospital_cost in 10.0
#               do
#                 for seed in 2345
#                 do
#                   C:/Users/kkris/Documents/GitHub/sb3-seir2/my_scripts/batch.sh $test_name $env_id $weight $Senario $health_cost_scale $rho_per_week $hospital_beds_ratio $max_hospital_cost $seed
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done