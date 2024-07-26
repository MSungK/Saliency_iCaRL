#!/usr/bin/zsh
set -e
max_process=4
cur_process=0

for beta in 1e-5 3e-5 5e-5 7e-5 9e-5 1e-4 3e-4 5e-4 7e-4 9e-4 1e-3 3e-3 5e-3 7e-3 9e-3 3e-2 5e-2 7e-2 9e-2 1e-1 3e-1 5e-1 
do  
    if [ ${cur_process} -ge ${max_process} ]; then
        wait -n
        cur_process=$((${cur_process} - 1))
    fi
    echo ${beta}
    python main.py \
        --beta ${beta} \
        --output ${beta} &
    cur_process=$((${cur_process} + 1))
done