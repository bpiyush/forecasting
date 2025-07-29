#!/bin/bash

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <start_index> <end_index>"
    echo "Example: $0 0 100"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2

# Validate input
if [ $START_INDEX -lt 0 ] || [ $END_INDEX -lt $START_INDEX ]; then
    echo "Error: Invalid indices. Start index must be >= 0 and end index must be >= start index"
    exit 1
fi

echo "Processing clips from index $START_INDEX to $END_INDEX across 4 GPUs"

# Set up directories and paths
EGO4D_CLIP_DIR=/scratch/shared/beegfs/shared-datasets/EGO4D/ego4d_data_v1/clips/
TARGET_DIR=/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/clips_360p/
mkdir -p ${TARGET_DIR}

# Get all clips and sort them
readarray -d '' CLIPS < <(find ${EGO4D_CLIP_DIR} -name "*.mp4" -print0)
IFS=$'\n' CLIPS=($(sort <<<"${CLIPS[*]}"))
unset IFS

# Calculate total clips to process
TOTAL_CLIPS=$((END_INDEX - START_INDEX + 1))
CLIPS_PER_GPU=$((TOTAL_CLIPS / 4))
REMAINDER=$((TOTAL_CLIPS % 4))

echo "Total clips to process: $TOTAL_CLIPS"
echo "Clips per GPU: $CLIPS_PER_GPU"
echo "Remainder: $REMAINDER"

# Function to process clips for a specific GPU
process_gpu_clips() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    
    echo "GPU $gpu_id: Processing clips $start_idx to $end_idx"
    
    # Set GPU device
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    for ((i=start_idx; i<=end_idx; i++)); do
        local clip_index=$((START_INDEX + i))
        
        # Check if the index is valid
        if [ ${clip_index} -ge ${#CLIPS[@]} ]; then
            echo "GPU $gpu_id: Error: Clip index ${clip_index} is out of range. Total clips: ${#CLIPS[@]}"
            continue
        fi
        
        local clip=${CLIPS[${clip_index}]}
        local filename=$(basename "$clip")
        
        echo "GPU $gpu_id: Processing clip ${clip_index}: ${filename}"
        
        ffmpeg -y -i "$clip" \
            -c:v libx264 \
            -crf 28 \
            -vf "scale=320:320:force_original_aspect_ratio=increase,pad='iw+mod(iw,2)':'ih+mod(ih,2)'" \
            -an "${TARGET_DIR}/${filename}"
    done
    
    echo "GPU $gpu_id: Completed processing"
}

# Calculate start and end indices for each GPU
GPU0_START=0
GPU0_END=$((CLIPS_PER_GPU - 1))

GPU1_START=$CLIPS_PER_GPU
GPU1_END=$((2 * CLIPS_PER_GPU - 1))

GPU2_START=$((2 * CLIPS_PER_GPU))
GPU2_END=$((3 * CLIPS_PER_GPU - 1))

GPU3_START=$((3 * CLIPS_PER_GPU))
GPU3_END=$((4 * CLIPS_PER_GPU - 1))

# Distribute remainder to first few GPUs
if [ $REMAINDER -gt 0 ]; then
    GPU0_END=$((GPU0_END + 1))
    if [ $REMAINDER -gt 1 ]; then
        GPU1_END=$((GPU1_END + 1))
    fi
    if [ $REMAINDER -gt 2 ]; then
        GPU2_END=$((GPU2_END + 1))
    fi
fi

echo "GPU 0: clips $GPU0_START to $GPU0_END"
echo "GPU 1: clips $GPU1_START to $GPU1_END"
echo "GPU 2: clips $GPU2_START to $GPU2_END"
echo "GPU 3: clips $GPU3_START to $GPU3_END"

# Start processing on all GPUs in parallel
process_gpu_clips 0 $GPU0_START $GPU0_END &
process_gpu_clips 1 $GPU1_START $GPU1_END &
process_gpu_clips 2 $GPU2_START $GPU2_END &
process_gpu_clips 3 $GPU3_START $GPU3_END &

# Wait for all background processes to complete
wait

echo "All GPU processing completed!" 