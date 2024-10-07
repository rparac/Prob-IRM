#!/bin/bash

job_ids=(66073 66074 66075 66076 66077 66078 66079 66080 66081 66082 66083 66084 66085 66086 66087 66088 66089 66090 66091 66092)



# Loop through each job ID and cancel it
for job_id in ${job_ids[@]}; do
    echo $job_id
    qdel $job_id
done
