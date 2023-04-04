#!/bin/bash


for nhi in 6.0
    do
    for freq in 353
        do
        for noisety in IIcib
             do
	     echo $freq
               python template_produce_v10.py 512 32 $freq 0.12 $noisety $nhi	     
	          python CIB_HII_variance/compute_CIB_variance_v4.py 512 32 $freq $noisety PR3 $nhi
               python main_Bayesian_code_Sim_Test_V9.py 512 32 $freq 5000 0.4 $noisety 0.05 0.5 1 $nhi
        done
     done
done
