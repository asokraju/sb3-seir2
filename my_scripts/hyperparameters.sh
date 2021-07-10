#!/bin/bash -l

for test_name in test_grace_6
do
  for env_id in gym_seir:seir-b-v0
  do
    for weight in 0.5
    do
      for Senario in 1
      do
        for health_cost_scale in 581.0
        do
          for rho_per_week in 0.02
          do
            for hospital_beds_ratio in 0.00287
            do
              for max_hospital_cost in 10 
              do
                for seed in  2532 1079 2774 919
                do
                  C:/Users/k/Documents/GitHub/sb3-seir2/my_scripts/batch.sh $test_name $env_id $weight $Senario $health_cost_scale $rho_per_week $hospital_beds_ratio $max_hospital_cost $seed
                done
              done
            done
          done
        done
      done
    done
  done
done