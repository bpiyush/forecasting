#!/bin/bash

# Submit multiple jobs to process clips from 1720 to 5000
# Each job will process 500 clips to stay within SLURM limits

START_INDEX=1720
END_INDEX=5000
BATCH_SIZE=500

for ((start=$START_INDEX; start<=$END_INDEX; start+=$BATCH_SIZE)); do
    end=$((start + BATCH_SIZE - 1))
    if [ $end -gt $END_INDEX ]; then
        end=$END_INDEX
    fi
    
    echo "Submitting job for indices $start-$end"
    
    # Create a temporary script with the correct array range
    cat > temp_resize_script.sh << EOF
#!/bin/bash
#SBATCH --job-name=resize_ego4d_clips_${start}_${end}
#SBATCH --array=${start}-${end}
#SBATCH --output=slurm/log_%A_%a.out
#SBATCH --error=slurm/log_%A_%a.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem=10GB
#SBATCH --time=1:00:00
#SBATCH --signal=USR1@600
#SBATCH --partition=gpu
#SBATCH --requeue

source ~/.bashrc
cd /users/piyush/projects/forecasting

EGO4D_CLIP_DIR=/scratch/shared/beegfs/shared-datasets/EGO4D/ego4d_data_v1/clips/
TARGET_DIR=/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/clips_360p/
mkdir -p \${TARGET_DIR}

readarray -d '' CLIPS < <(find \${EGO4D_CLIP_DIR} -name "*.mp4" -print0)
IFS=\$'\n' CLIPS=(\$(sort <<<"\${CLIPS[*]}"))
unset IFS

CLIP=\${CLIPS[\${SLURM_ARRAY_TASK_ID}]}
FILENAME=\`basename \$CLIP\`

ffmpeg -y -i \$CLIP \\
	-c:v libx264 \\
	-crf 28 \\
	-vf "scale=320:320:force_original_aspect_ratio=increase,pad='iw+mod(iw,2)':'ih+mod(ih,2)'" \\
	-an \${TARGET_DIR}/\${FILENAME}
EOF

    # Submit the job
    sbatch temp_resize_script.sh
    
    # Clean up
    rm temp_resize_script.sh
    
    echo "Submitted job for indices $start-$end"
done

echo "All jobs submitted!" 