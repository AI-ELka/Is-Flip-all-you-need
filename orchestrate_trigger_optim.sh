#!/bin/bash
set -e
set -x

# ============================================================
#  Orchestrate trigger optimisation across remote machines.
#  One job per (dataset, aggregator) — no gen_labels, no train_user.
#
#  Usage:
#    chmod +x orchestrate_trigger_optim.sh
#    ./orchestrate_trigger_optim.sh
# ============================================================

BASE_DIR="$HOME/Desktop/Is-Flip-all-you-need"
LOG_DIR="$BASE_DIR/logs/trigger_optim_withnorm_ssim"
mkdir -p "$LOG_DIR"

# ---- Experiment grid (must match gen_config_trigger.py) ----
DATASETS=("cifar")        # add "svhn" etc. as needed
AGGREGATORS=("mean" "median" "krum" "trmean" "multikrum")
NUM_CLEAN=7
NUM_POISONED=3
MODEL_FLAG="r32p"
POISONER="optimized"

CONFIG_DIR="federated_experiments_withnorm"
# ---- Machine pool ----
MACHINES=(
    # "poly-acromion"
    # "poly-apophyse"
    # "poly-astragale"
    # "poly-atlas"
    # "poly-axis"
    # "poly-coccyx"
    # "poly-cote"
    # "poly-cubitus"
    # "poly-cuboide"
    # "poly-femur"
    # "poly-frontal"
    # "poly-humerus"
    # "poly-malleole"
    # "poly-bengali" 
    # "poly-coucou" 
    # "poly-dindon" 
    # "poly-epervier" 
    # "poly-faisan" 
    # "poly-gelinotte" 
    # "poly-hibou" 
    # "poly-harpie" 
    # "poly-jabiru" 
    # "poly-kamiche" 
    # "poly-linotte" 
    # "poly-loriol" 
    # "poly-mouette" 
    # "poly-nandou" 
    # "poly-ombrette" 
    "poly-perdrix" 
    "poly-quetzal" 
    "poly-quiscale" 
    "poly-rouloul" 
    "poly-sitelle" 
    "poly-traquet" 
    "poly-urabu" 
    "poly-verdier"
)
N_MACHINES=${#MACHINES[@]}

# ---- Helpers (same pattern as orchestrate_runs.sh) ----
run_remote() {
    local machine=$1
    local cmd=$2
    local done_file=$3
    local log_file=$4

    echo "[LAUNCH] $machine → $cmd"
    ssh "$machine" "
        cd $BASE_DIR &&
        source /users/eleves-b/2022/abdessamad.el-kabid/Desktop/.venv/bin/activate &&
        nohup bash -c '$cmd; touch $done_file' > $log_file 2>&1 &
    "
}

wait_for_done_files() {
    local files=("$@")
    echo "[WAIT] Waiting for trigger optimisation jobs..."
    while true; do
        all_done=true
        for f in "${files[@]}"; do
            [ ! -f "$f" ] && all_done=false && break
        done
        $all_done && break
        sleep 10
    done
    echo "[DONE] All trigger optimisation jobs finished."
}

# ---- Clean previous logs ----
echo "Cleaning previous trigger-optim logs..."
rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.done || true

# ---- Build job list ----
JOBS=()
for dataset in "${DATASETS[@]}"; do
    for aggregator in "${AGGREGATORS[@]}"; do
        config="${CONFIG_DIR}/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${dataset}/backdoor/${aggregator}/${POISONER}/trigger"
        JOBS+=("$config")
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo "Total trigger-optim jobs: $TOTAL_JOBS"

# ---- Dispatch ----
if [ "$TOTAL_JOBS" -le "$N_MACHINES" ]; then
    # Few jobs: launch all at once and wait
    DONE_FILES=()
    for i in "${!JOBS[@]}"; do
        config="${JOBS[$i]}"
        machine="${MACHINES[$((i % N_MACHINES))]}"
        safe_name="trig_$(echo "$config" | tr '/' '_')"
        done_file="$LOG_DIR/${safe_name}.done"
        log_file="$LOG_DIR/${safe_name}.log"
        rm -f "$done_file"

        run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &
        DONE_FILES+=("$done_file")
    done

    wait_for_done_files "${DONE_FILES[@]}"
else
    # More jobs than machines: use a pool scheduler
    INDEX=0
    declare -a running
    for ((i=0; i<N_MACHINES; i++)); do running[i]=""; done

    while true; do
        # Assign jobs to free machines
        for ((i=0; i<N_MACHINES; i++)); do
            if [ -z "${running[i]:-}" ] && [ $INDEX -lt $TOTAL_JOBS ]; then
                config="${JOBS[$INDEX]}"
                machine="${MACHINES[$i]}"
                safe_name="trig_$(echo "$config" | tr '/' '_')"
                done_file="$LOG_DIR/${safe_name}.done"
                log_file="$LOG_DIR/${safe_name}.log"
                rm -f "$done_file"

                run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &
                running[i]=$done_file
                INDEX=$((INDEX + 1))
                echo "[POOL] $machine ← job $INDEX/$TOTAL_JOBS ($config)"
            fi
        done

        # Check if everything is done
        all_idle=true
        for ((i=0; i<N_MACHINES; i++)); do
            [ -n "${running[i]:-}" ] && all_idle=false && break
        done
        if $all_idle && [ $INDEX -ge $TOTAL_JOBS ]; then
            break
        fi

        # Wait for at least one machine to finish
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
fi

echo "ALL TRIGGER OPTIMISATION DONE"


# or do 