usage() {
  echo "Usage: ${0} [--glove] [--gpu_id]" 1>&2
  exit 1 
}
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
    *)
      usage
      shift
      ;;
  esac
done

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    --do-train \
    --do-eval