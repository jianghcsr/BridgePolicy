# bash scripts/gen_demonstration_adroit.sh door
# bash scripts/gen_demonstration_adroit.sh hammer
# bash scripts/gen_demonstration_adroit.sh pen

cd third_party/VRL3/src

# task=${1}
task="hammer"
CUDA_VISIBLE_DEVICES=0 python gen_demonstration_expert.py --env_name $task \
                        --num_episodes 40 \
                        --root_dir "${BRIDGE_POLICY_DATA_ROOT:-__BRIDGE_POLICY_DATA_ROOT__}" \
                        --expert_ckpt_path "../vrl3_ckpts/vrl3_${task}.pt" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop


task="door"

CUDA_VISIBLE_DEVICES=0 python gen_demonstration_expert.py --env_name $task \
                        --num_episodes 40 \
                        --root_dir "${BRIDGE_POLICY_DATA_ROOT:-__BRIDGE_POLICY_DATA_ROOT__}" \
                        --expert_ckpt_path "../vrl3_ckpts/vrl3_${task}.pt" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop

