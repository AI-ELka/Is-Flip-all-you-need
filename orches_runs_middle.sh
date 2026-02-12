#!/bin/bash
set -e
set -x

############################################
# CONFIGURATION
############################################

BASE_DIR="$HOME/Desktop/Is-Flip-all-you-need"
LOG_DIR="$BASE_DIR/logs3"

mkdir -p "$LOG_DIR"

DATASET="cifar"
ATTACK="backdoor"


AGGREGATORS=("mean") 

BUDGETS=(150 300 500 1000 1500 2000 2500 5000) #150 300 500 1000 1500 2000 2500 5000)

# ---  Define specific runs to resume ---
TARGET_RUN_IDS=(1 2 3 4 5)

NUM_CLEAN=7
NUM_POISONED=3

MACHINES=(
    "poly-acromion"
    "poly-apophyse"
    "poly-astragale"
    "poly-atlas"
    "poly-axis"
    "poly-coccyx"
    "poly-cote"
    "poly-cubitus"
    "poly-cuboide"
    "poly-femur"
    "poly-frontal"
    "poly-humerus"
    "poly-malleole"
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

# --- MODIFICATION: Commented out to preserve logs from Run 1 ---
# echo "🧹 Cleaning previous logs and done files..."
# rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.done || true

############################################
# MAIN LOOP
############################################

for aggregator in "${AGGREGATORS[@]}"; do
    echo "========================================"
    echo "AGGREGATOR: $aggregator"
    echo "========================================"

    #####################################
    # PHASE A — gen_labels (SKIPPED)
    #####################################
    
    echo "[PHASE A] gen_labels - SKIPPING (Already finished)"

    # Loop commented out as requested
    # DONE_FILES=()
    # JOB_ID=0
    # for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
    #    ...
    # done
    # wait_for_done_files "${DONE_FILES[@]}"

    #####################################
    # PHASE B — train_user (BATCHED)
    #####################################

    echo "[PHASE B] train_user (Runs: ${TARGET_RUN_IDS[*]})"

    # 1) Construire la liste des jobs SPECIFIQUES (2, 3, 4, 5)
    JOBS=()
    
    # --- MODIFICATION: Iterate over TARGET_RUN_IDS instead of seq N_CYCLES ---
    for run_id in "${TARGET_RUN_IDS[@]}"; do
        for budget in "${BUDGETS[@]}"; do
            JOBS+=("$run_id|$budget")
        done
    done

    TOTAL_JOBS=${#JOBS[@]}
    INDEX=0

    # 2) Lancer par batchs de N_MACHINES
    while [ $INDEX -lt $TOTAL_JOBS ]; do
        DONE_FILES=()

        echo "[BATCH] Launching jobs $INDEX → $((INDEX + N_MACHINES - 1))"

        for ((i=0; i<N_MACHINES && INDEX<TOTAL_JOBS; i++)); do
            IFS='|' read -r run_id budget <<< "${JOBS[$INDEX]}"
            machine=${MACHINES[$i]}

            config="federated_experiments/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/train_user_${budget}/${run_id}"

            safe_name="train_${aggregator}_${run_id}_${budget}_${machine}"
            done_file="$LOG_DIR/${safe_name}.done"
            log_file="$LOG_DIR/${safe_name}.log"
            
            # Remove only the specific done file for this job, not everything
            rm -f "$done_file"

            run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &

            DONE_FILES+=("$done_file")
            INDEX=$((INDEX + 1))
        done

        wait_for_done_files "${DONE_FILES[@]}"
        echo "✅ Batch completed"
    done

    echo "✅ train_user all runs done"
done

echo "🎉 ALL DONE"