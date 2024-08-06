#!/bin/bash

# Cancels all jobs on a PBS cluster

# Get a list of your job IDs
job_ids=$(qstat -u rp218 | grep rp218 | awk '{print $1}' | cut -d. -f1)

# Loop through each job ID and cancel it
for job_id in $job_ids; do
    echo $job_id
    qdel $job_id
done

