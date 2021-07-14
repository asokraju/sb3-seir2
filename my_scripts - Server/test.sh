#!/bin/bash -l

seeds=(1259 133 975 1632 1108 2798 2000 2970 637 1633 456 65 245)

typeset -i variable=$(cat 'Bindex.txt')

seeds=${seeds[@]:$variable}

for seed in $seeds
	do
	variable=$((variable+1))
	echo $seed
	echo $variable > 'Bindex.txt' 
	done
	