#!/bin/bash
set -x
set -e

############################################
# CONFIGURATION
############################################

DATASET="cifar"
#POISONERS=("1xs" "1xp" "4xl")
AGGREGATOR=("mean" "median")
BUDGETS=(150 300 500 1000 2000 2500 5000 10000)
N_CYCLES=10
NUM_CLEAN=2
NUM_POISONED=1
ATTACK="backdoor"

BASE_DIR="$HOME/FLIP"
LOG_DIR="$BASE_DIR/logs"

MACHINES=(
bentley bugatti cadillac chrysler corvette ferrari fiat ford jaguar lada
maserati nissan niva peugeot pontiac rolls rover
royce simca skoda venturi volvo renault porsche
)

############################################
# INIT
############################################

mkdir -p "$LOG_DIR"
N_MACHINES=${#MACHINES[@]}
JOB_ID=0

############################################
# FUNCTIONS
############################################

run_remote() {
    local machine=$1
    local cmd=$2
    local done_file=$3
    local log=$4

    echo "[LAUNCH] $machine → $cmd"

    ssh "$machine" "
        cd $BASE_DIR &&
        nohup bash -c '$cmd; touch $done_file' > $log 2>&1 &
    "
}

wait_for_done_files() {
    local files=("$@")
    echo "[WAIT] Waiting for jobs to finish..."
    while true; do
        all_done=true
        for f in "${files[@]}"; do
            if [ ! -f "$f" ]; then
                all_done=false
                break
            fi
        done
        $all_done && break
        sleep 10
    done
    echo "[DONE] Phase completed"
}

############################################
# MAIN LOOP
############################################

for cycle in $(seq 1 $N_CYCLES); do
    echo
    echo "=========================================="
    echo "        CYCLE $cycle / $N_CYCLES"
    echo "=========================================="

    ########################################
    # PHASE A — poisoner init (PARALLEL)
    ########################################
    echo "[PHASE A] Initial poisoner runs"
    DONE_FILES=()
    JOB_CMDS=()

    # Préparer tous les jobs
    for aggregator in "${AGGREGATOR[@]}"; do
        JOB_CMDS+=("python run_experiment.py federated_experiments/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/gen_labels")
    done

    JOB_ID=0
    for cmd in "${JOB_CMDS[@]}"; do
        machine=${MACHINES[$((JOB_ID % N_MACHINES))]}

        safe_cmd=${cmd//[ \/]/_}

        done_file="$LOG_DIR/cycle${cycle}_init_${safe_cmd}.done"
        log="$LOG_DIR/cycle${cycle}_init_${safe_cmd}.log"
        rm -f "$done_file"

        run_remote "$machine" "$cmd" "$done_file" "$log" &

        DONE_FILES+=("$done_file")
        JOB_ID=$((JOB_ID + 1))
    done

    wait_for_done_files "${DONE_FILES[@]}"

    ########################################
    # PHASE B — training (PARALLEL)
    ########################################
    echo "[PHASE B] Training runs"
    DONE_FILES=()
    JOB_CMDS=()

    for aggregator in "${AGGREGATOR[@]}"; do
        for budget in "${BUDGETS[@]}"; do
            JOB_CMDS+=("python run_experiment.py federated_experiments/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/train_user_${budget}")
        done
    done

    JOB_ID=0
    for cmd in "${JOB_CMDS[@]}"; do
        machine=${MACHINES[$((JOB_ID % N_MACHINES))]}
        safe_cmd=${cmd//[ \/]/_}
        done_file="$LOG_DIR/cycle${cycle}_train_${safe_cmd}.done"
        log="$LOG_DIR/cycle${cycle}_train_${safe_cmd}.log"
        rm -f "$done_file"

        run_remote "$machine" "$cmd" "$done_file" "$log" &

        DONE_FILES+=("$done_file")
        JOB_ID=$((JOB_ID + 1))
    done

    wait_for_done_files "${DONE_FILES[@]}"

    ########################################
    # PHASE C — update
    ########################################
    echo "[PHASE C] update_configs.py"
    cd "$BASE_DIR"
    python update_configs.py | tee "$LOG_DIR/cycle${cycle}_update.log"

done

echo
echo "🎉 ALL CYCLES COMPLETED SUCCESSFULLY"
