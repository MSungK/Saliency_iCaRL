#!/usr/bin/zsh
set -e
max_process=4
cur_process=0

for beta in 1e-5 1e-6 1e-7 1e-8
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