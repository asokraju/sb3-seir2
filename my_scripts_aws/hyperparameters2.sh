#!/bin/bash -l
for test_name in test_yang_4_correction
do
  for env_id in gym_seir:seir-b-v0
  do
    for weight in 0.5
    do
      for Senario in 1 #0 2
      do
        for health_cost_scale in 581.0 #600.0
        do
          for rho_per_week in 0.02
          do
            for hospital_beds_ratio in 0.01 #0.00287
            do
              for max_hospital_cost in 10.0 5.0
              do
                for seed in 2901 2345
                do
                  for learning_rate in 0.0003 0.0002
                  do
                    for clip_range in 0.1
                    do
                      for rl_algo in 0 1 
                      do
                        /home/ec2-user/sb3-seir2/my_scripts_aws/batch.sh $test_name $env_id $weight $Senario $health_cost_scale $rho_per_week $hospital_beds_ratio $max_hospital_cost $seed $learning_rate $clip_range $rl_algo
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