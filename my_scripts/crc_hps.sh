#!/bin/bash -l

for test_name in test_1
do
  for env_id in gym_seir:seir-b-v0
  do
    for weight in 0.5
    do
      for Senario in 0
      do
        for health_cost_scale in 600.0
        do
          for rho_per_week in 0.02 0.01 0.03
          do
            for hospital_beds_ratio in 0.005 0.001 0.01
            do
              for max_hospital_cost in 100.0 200.0
              do
                for seed in 2345
                do
                  for learning_rate in 0.0003 0.0002 0.0001
                  do
                    for clip_range in 0.1 0.15 0.2
                    do
                      /afs/crc.nd.edu/user/k/kkosaraj/sb3-seir2/scripts/crc_batch.sh $test_name $env_id $weight $Senario $health_cost_scale $rho_per_week $hospital_beds_ratio $max_hospital_cost $seed $learning_rate $clip_range
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