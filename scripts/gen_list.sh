#ENV_LIST='SparseMountainCarContinuous-v0 MountainCarContinuous-v0 HalfCheetah-v2 SparseHalfCheetah-v2 Reacher-v2 Hopper-v2 Humanoid-v2 Ant-v2 SparseInvertedPendulumEnv-v2 SparseInvertedDoublePendulum-v2 Walker2d-v2 Swimmer-v2'
SEED_LIST='2 3 5 7'

setting_num=0
setting_num=$[setting_num+1]; SETTING[$setting_num]='--SGLD-mode 2'


rm exp_list.txt
touch exp_list.txt

ENV_LIST='SparseMountainCarContinuous-v0 MountainCarContinuous-v0 SparseHalfCheetah-v2 SparseInvertedPendulumEnv-v2'

for env_name in $ENV_LIST
do
    exp_id=1
    for i in `seq 1 $setting_num`
    do
      seed_id=1
      for seed in $SEED_LIST
      do
        echo "python DDPG/trainer.py  --env $env_name  --exp-name ${exp_id}_${seed_id}  --rand-seed $seed ${SETTING[$i]} --train-mode 1" >> exp_list.txt
        seed_id=$[seed_id+1]
      done
      exp_id=$[exp_id+1]
    done 
done

setting_num=0
setting_num=$[setting_num+1]; SETTING[$setting_num]='--SGLD-mode 2'

ENV_LIST='HalfCheetah-v2 Reacher-v2 Hopper-v2 InvertedPendulum-v2'

for env_name in $ENV_LIST
do
    exp_id=1
    for i in `seq 1 $setting_num`
    do
      seed_id=1
      for seed in $SEED_LIST
      do
        echo "python DDPG/trainer.py  --env $env_name  --exp-name ${exp_id}_${seed_id}  --rand-seed $seed ${SETTING[$i]} --train-mode 1" >> exp_list.txt
        seed_id=$[seed_id+1]
      done
      exp_id=$[exp_id+1]
    done 
done


cp exp_list.txt results/

