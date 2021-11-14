usage() {
  echo "Usage: ${0} [--glove] [--gpu_id] [--mode] [--roberta] [--baseline] [--do_test] [--do_train]" 1>&2
  exit 1 
}

FLAG_train=0
FLAG_test=0

while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    --glove)
        GLOVE=${2}
        shift 2
        ;;
    --gpu_id)
        GPU=${2}
        shift 2
        ;;
    --mode)
        MODE=${2}
        shift 2
        ;;
    --roberta)
        ROBERTA_DIR=${2}
        shift 2
        ;;
    --baseline)
        BASELINE=${2}
        shift 2
        ;;
    --do_test)
        FLAG_test=1
        shift 1
        ;;
    --do_train)
        FLAG_train=1
        shift 1
        ;;
    *)
      usage
      shift
      ;;
  esac
done

WORK_DIR=$(cd $(dirname $(dirname $0));pwd)
if [ "${MODE}" == "soft" ]; then
    OUT_DIR=${WORK_DIR}/outputs/soft_gen
    mkdir -p ${OUT_DIR}
    WORK_DIR=${WORK_DIR}/soft_gen
else
    OUT_DIR=${WORK_DIR}/outputs/hard_gen
    mkdir -p ${OUT_DIR}
    WORK_DIR=${WORK_DIR}/hard_gen   
fi
cd ${WORK_DIR}


if [ ${FLAG_train} -eq 1 ]; then
    echo "begin train"
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
        --do-train \
        --do-eval \
        --glove ${GLOVE} \
        --emotion-model ../outputs/emotion/best_emotion.pt \
        --bert-score-baseline ${BASELINE} \
        --bert-score-model ${ROBERTA_DIR} \
        --output-dir ${OUT_DIR}
fi

if [ ${FLAG_test} -eq 1 ]; then
    echo "begin test"
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
        --do-test \
        --glove ${GLOVE} \
        --emotion-model ../outputs/emotion/best_emotion.pt \
        --checkpoint ${OUT_DIR}/checkpoint_best.pt \
        --bert-score-baseline ${BASELINE} \
        --bert-score-model ${ROBERTA_DIR} \
        --output-dir ${OUT_DIR} \
        --log-file result.txt
fi



# bash bash/run_generation.sh --glove /home/liuyuhan/datasets/glove/glove.6B.300d.txt --gpu_id 2 --mode soft --roberta /home/liuyuhan/datasets/roberta-large-en/ --baseline /home/liuyuhan/datasets/roberta-large-en/roberta-large.tsv


