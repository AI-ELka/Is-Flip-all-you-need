#!/bin/bash
set -e
set -x

############################################
# CONFIGURATION
############################################

BASE_DIR="$HOME/FLIP"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

DATASET="cifar"
ATTACK="backdoor"
AGGREGATORS=("mean" "median" "krum" "trmean")
BUDGETS=(150 300 500 1000 1500 2000 2500 5000)
N_CYCLES=10
NUM_CLEAN=6
NUM_POISONED=4

MACHINES=(
bentley bugatti cadillac chrysler corvette ferrari fiat ford jaguar lada
maserati nissan niva peugeot pontiac rolls rover
royce simca skoda venturi volvo renault porsche
)

N_MACHINES=${#MACHINES[@]}

############################################
# FUNCTIONS
############################################

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

############################################
# CLEANUP
############################################

echo "🧹 Cleaning previous logs and done files..."
rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.done || true

############################################
# MAIN
############################################

JOB_ID=0

for aggregator in "${AGGREGATORS[@]}"; do
    echo "========================================"
    echo "AGGREGATOR: $aggregator"
    echo "========================================"

    #####################################
    # PHASE A — gen_labels (PARALLEL)
    #####################################

    echo "[PHASE A] gen_labels"

    DONE_FILES=()
    JOB_ID=0

    for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
        machine=${MACHINES[$((JOB_ID % N_MACHINES))]}

        config="federated_experiments/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/gen_labels/${run_id}"

        safe_name="gen_${aggregator}_${run_id}_${machine}"
        done_file="$LOG_DIR/${safe_name}.done"
        log_file="$LOG_DIR/${safe_name}.log"
        rm -f "$done_file"

        run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &

        DONE_FILES+=("$done_file")
        JOB_ID=$((JOB_ID + 1))
    done

    wait_for_done_files "${DONE_FILES[@]}"
    echo "✅ gen_labels done"

    #####################################
    # PHASE B — train_user (1 job max par machine)
    #####################################

    for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
        echo "[PHASE B] train_user — run $run_id"

        # Préparer tous les jobs
        JOBS=()
        for budget in "${BUDGETS[@]}"; do
            JOBS+=("$budget")
        done

        # Tant qu'il reste des jobs à assigner
        while [ ${#JOBS[@]} -gt 0 ]; do
            NEW_JOBS=()
            for budget in "${JOBS[@]}"; do
                # Chercher une machine libre
                machine=""
                for m in "${MACHINES[@]}"; do
                    done_file="$LOG_DIR/train_${aggregator}_${run_id}_${budget}_${m}.done"
                    if [ ! -f "$done_file" ]; then
                        machine="$m"
                        break
                    fi
                done

                if [ -n "$machine" ]; then
                    # Lancer le job sur la machine libre
                    config="federated_experiments/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/train_user_${budget}/${run_id}"
                    safe_name="train_${aggregator}_${run_id}_${budget}_${machine}"
                    done_file="$LOG_DIR/${safe_name}.done"
                    log_file="$LOG_DIR/${safe_name}.log"
                    rm -f "$done_file"

                    run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &
                    echo "🚀 Started train_user: $aggregator, budget $budget, run $run_id on $machine"
                else
                    # Machine occupée, job reporté
                    NEW_JOBS+=("$budget")
                fi
            done

            JOBS=("${NEW_JOBS[@]}")
            sleep 5
        done

        # Attendre tous les done pour ce run
        DONE_FILES=()
        for budget in "${BUDGETS[@]}"; do
            for machine in "${MACHINES[@]}"; do
                done_file="$LOG_DIR/train_${aggregator}_${run_id}_${budget}_${machine}.done"
                DONE_FILES+=("$done_file")
            done
        done

        wait_for_done_files "${DONE_FILES[@]}"
        echo "✅ train_user run $run_id done"
    done
done

echo "🎉 ALL DONE"
