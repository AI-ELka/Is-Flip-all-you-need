#!/bin/bash
set -e
set -x

BASE_DIR="$HOME/FLIP"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

DATASET="cifar"
ATTACK="backdoor"
AGGREGATORS=("mean" "median" "krum" "trmean")
BUDGETS=(150 300 500 1000 1500 2000 2500 5000)
N_CYCLES=5
NUM_CLEAN=6
NUM_POISONED=4
MODEL_FLAG="r32p"
POISONER="1xs"

MACHINES=(
bentley bugatti cadillac chrysler corvette ferrari ford jaguar lada
maserati nissan niva peugeot pontiac rolls rover
royce simca skoda venturi volvo renault porsche fiat
)

N_MACHINES=${#MACHINES[@]}

run_remote() {
    local machine=$1
    local cmd=$2
    local done_file=$3
    local log_file=$4

    echo "[LAUNCH] $machine → $cmd"

    ssh "$machine" "
        cd $BASE_DIR &&
        nohup bash -c '$cmd; touch $done_file' > $log_file 2>&1 &
    "
}

wait_for_done_files() {
    local files=("$@")
    echo "[WAIT] Waiting for jobs to finish..."
    while true; do
        all_done=true
        for f in "${files[@]}"; do
            [ ! -f "$f" ] && all_done=false && break
        done
        $all_done && break
        sleep 10
    done
    echo "[DONE] Phase completed"
}

echo "Cleaning previous logs and done files..."
rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.done || true

for aggregator in "${AGGREGATORS[@]}"; do
    echo "========================================"
    echo "AGGREGATOR: $aggregator"
    echo "========================================"

    echo "1 - gen_labels"

    DONE_FILES=()
    JOB_ID=0

    for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
        machine=${MACHINES[$((JOB_ID % N_MACHINES))]}

        config="federated_experiments/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/${POISONER}/gen_labels/${run_id}"

        safe_name="gen_${MODEL_FLAG}_${NUM_POISONED}vs${NUM_CLEAN}_${DATASET}_${ATTACK}_${aggregator}_${POISONER}_${run_id}_${machine}"
        done_file="$LOG_DIR/${safe_name}.done"
        log_file="$LOG_DIR/${safe_name}.log"
        rm -f "$done_file"

        run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &

        DONE_FILES+=("$done_file")
        JOB_ID=$((JOB_ID + 1))
    done

    wait_for_done_files "${DONE_FILES[@]}"
    echo "gen_labels done"

    echo "2 - train_user"

    JOBS=()
    for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
        for budget in "${BUDGETS[@]}"; do
            JOBS+=("$run_id|$budget")
        done
    done

    TOTAL_JOBS=${#JOBS[@]}
    INDEX=0

    echo "Total jobs: $TOTAL_JOBS"

    declare -a running
    for ((i=0; i<N_MACHINES; i++)); do running[i]=""; done

    while true; do

        # Assign jobs to free machines
        for ((i=0; i<N_MACHINES; i++)); do

            # If machine is not running and there are still jobs to run
            if [ -z "${running[i]:-}" ] && [ $INDEX -lt $TOTAL_JOBS ]; then
                IFS='|' read -r run_id budget <<< "${JOBS[$INDEX]}"
                machine=${MACHINES[$i]}

                config="federated_experiments/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/${POISONER}/train_user_${budget}/${run_id}"

                safe_name="${MODEL_FLAG}_${NUM_POISONED}vs${NUM_CLEAN}_${DATASET}_${ATTACK}_${aggregator}_${POISONER}_${run_id}_${machine}"
                done_file="$LOG_DIR/${safe_name}.done"
                log_file="$LOG_DIR/${safe_name}.log"
                rm -f "$done_file"

                run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &
                running[i]=$done_file
                INDEX=$((INDEX + 1))
                echo "[POOL] $machine ← job $INDEX/$TOTAL_JOBS"
            fi
        done


        # Break the loop if all machines are idle and there are no more jobs to run
        all_idle=true
        for ((i=0; i<N_MACHINES; i++)); do
            [ -n "${running[i]:-}" ] && all_idle=false && break
        done
        if $all_idle && [ $INDEX -ge $TOTAL_JOBS ]; then
            break
        fi

        # Wait at least one machine to finish, then asing a new job
        while true; do
            for ((i=0; i<N_MACHINES; i++)); do
                if [ -n "${running[i]:-}" ] && [ -f "${running[i]}" ]; then
                    running[i]=""
                    break 2
                fi
            done
            sleep 5
        done

    done

    echo "train_user all runs done"

    echo "Cleaning previous logs and done files..."
    rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.done || true

done

echo "ALL DONE"
