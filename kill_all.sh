#!/bin/bash

BASE_DIR="$HOME/FLIP"

MACHINES=(
bentley bugatti cadillac chrysler corvette ferrari fiat ford jaguar lada
maserati nissan niva peugeot pontiac rolls rover
royce simca skoda venturi volvo renault porsche
)

for m in "${MACHINES[@]}"; do
    echo "==== $m ===="

    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$m" "echo ok" &>/dev/null; then
        ssh "$m" "
            pkill -f 'python run_experiment.py' || true
            pkill -f 'bash -c python run_experiment.py' || true
            pkill -f '$BASE_DIR' || true
        "
        echo "✔ Killed on $m"
    else
        echo "❌ Cannot SSH to $m"
    fi
done

echo "🔥 Kill done."
