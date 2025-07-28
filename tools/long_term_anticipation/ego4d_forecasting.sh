function run(){
    
  NAME=$1
  CONFIG=$2
  shift 2;

  python -m scripts.run_lta \
    --job_name $NAME \
    --working_directory ${WORK_DIR} \
    --cfg $CONFIG \
    ${CLUSTER_ARGS} \
    DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS} \
    DATA.PATH_PREFIX ${EGO4D_VIDEOS} \
    CHECKPOINT_LOAD_MODEL_HEAD False \
    MODEL.FREEZE_BACKBONE True \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    FORECASTING.AGGREGATOR "" \
    FORECASTING.DECODER "" \
    $@
}

#-----------------------------------------------------------------------------------------------#

WORK_DIR=$1
if [ -z "${WORK_DIR}" ]; then
    WORK_DIR=/work/piyush/experiments/ego4d_forecasting/
fi
mkdir -p ${WORK_DIR}


DATA_ROOT=/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap
# EGO4D_ANNOTS=${DATA_ROOT}/long_term_anticipation/annotations/
EGO4D_ANNOTS=${DATA_ROOT}/long_term_anticipation/annotations_debug/
EGO4D_VIDEOS=${DATA_ROOT}/clips_360p/

# To run on dev machine: Single node configuration with 4 GPUs
CLUSTER_ARGS="NUM_GPUS 4 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 8"

# SlowFast-Transformer
BACKBONE_WTS=${DATA_ROOT}/long_term_anticipation/lta_models/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt
run slowfast_trf \
    configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml \
    FORECASTING.AGGREGATOR TransformerAggregator \
    FORECASTING.DECODER MultiHeadDecoder \
    FORECASTING.NUM_INPUT_CLIPS 4 \
    DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

#-----------------------------------------------------------------------------------------------#
#                                    OTHER OPTIONS                                              #
#-----------------------------------------------------------------------------------------------#

# # SlowFast-Concat
# BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt
# run slowfast_concat \
#     configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml \
#     FORECASTING.AGGREGATOR ConcatAggregator \
#     FORECASTING.DECODER MultiHeadDecoder \
#     DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

# # MViT-Concat
# BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/ego4d_mvit16x4.ckpt
# run mvit_concat \
#     configs/Ego4dLTA/MULTIMVIT_16x4.yaml \
#     FORECASTING.AGGREGATOR ConcatAggregator \
#     FORECASTING.DECODER MultiHeadDecoder \
#     DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

# # Debug locally using a smaller batch size / fewer GPUs
# CLUSTER_ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 32"

