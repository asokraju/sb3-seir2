#!/bin/bash

TEST_NAME=$1
ENV_ID=$2
WEIGHT=$3
SENARIO=$4
HCS=$5
RPW=$6
HBR=$7
MHC=$8
SEED=$9
#Creating necessary folders to save the results of the experiment
PARENT_DIR=$PWD  #"$(dirname $PWD)"             # file is inside this  directory
EXEC_DIR=$PWD                            # batch script is inside this dir
TEST_NAME_DIR="test_name=${TEST_NAME}"   # directory with test name

SEED_DIR="seed=${SEED}"
WEIGHT_DIR="w=${WEIGHT}"
SENARIO_DIR="w=${SENARIO}"
HCS_DIR="hcs=${HCS}"
RPW_DIR="rpw=${RPW}"
HBR_DIR="hbr=${HBR}"
MHC_DIR="mhc=${MHC}"

mkdir -p $TEST_NAME_DIR                  #making a directory with test name
RESULTS_DIR=${EXEC_DIR}/${TEST_NAME_DIR} # Directory for results
cd $RESULTS_DIR
                          #we are inside the results_dir
mkdir -p $WEIGHT_DIR                     #making a directory for parameter seed name
cd $WEIGHT_DIR

mkdir -p $SEED_DIR                       #making a directory for parameter weight name
cd $SEED_DIR

mkdir -p $SENARIO_DIR                    #making a directory for parameter Senario name
cd $SENARIO_DIR

mkdir -p $HCS_DIR                    
cd $HCS_DIR

mkdir -p $RPW_DIR                     
cd $RPW_DIR

mkdir -p $HBR_DIR                     
cd $HBR_DIR

mkdir -p $MHC_DIR                     
cd $MHC_DIR


export run_exec=$PARENT_DIR/argparse_run.py #python script that we want to run
$env_id $weight $Senario $health_cost_scale $rho_per_week $hospital_beds_ratio $max_hospital_cost
export run_flags="--env_id=${ENV_ID} --weight=${WEIGHT} --seed=${SEED} --Senario=${SENARIO} --health_cost_scale=${HCS} --rho_per_week=${RPW} --hospital_beds_ratio=${HBR} --max_hospital_cost=${MHC} --summary_dir=$PWD"
C:/Users/kkris/anaconda3/envs/sbl3/python.exe $run_exec $run_flags