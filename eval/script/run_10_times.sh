#!/bin/bash
set -euo pipefail

PREV=""

for REPEAT in $(seq 1 27); do
    if [[ -z "$PREV" ]]; then
        JOBID=$(sbatch \
            --parsable \
            --export=ALL,REPEAT_ID="$REPEAT" \
            run_amd_single.sbatch)
    else
        JOBID=$(sbatch \
            --parsable \
            --dependency=afterany:"$PREV" \
            --export=ALL,REPEAT_ID="$REPEAT" \
            run_amd_single.sbatch)
    fi

    echo "Repeat $REPEAT submitted: $JOBID"
    PREV="$JOBID"
done
