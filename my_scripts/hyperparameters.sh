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
                  C:/Users/kkris/Documents/GitHub/sb3-seir2/my_scripts/batch.sh $test_name $env_id $weight $Senario $health_cost_scale $rho_per_week $hospital_beds_ratio $max_hospital_cost $seed
                done
              done
            done
          done
        done
      done
    done
  done
done