# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh bridge_policy_base adroit_hammer 0322 0 0



DEBUG=False
wandb_mode="${WANDB_MODE:-disabled}"
save_ckpt="${SAVE_CKPT:-False}"

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_root="${BRIDGE_POLICY_RUN_ROOT:-data/outputs}"
run_dir="${run_root}/${exp_name}_seed${seed}"

gpu_id=${5}


cd BridgePolicy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python eval.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                
