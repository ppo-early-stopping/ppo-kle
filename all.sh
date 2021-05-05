for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Reacher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Pusher-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Thrower-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

wait

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Striker-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id InvertedPendulum-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id HalfCheetah-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Hopper-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

wait

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Swimmer-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Walker2d-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Ant-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 4 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 10 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 20 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 40 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-rollback-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-rollback --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

wait

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.015 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.015 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.02 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.02 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.025 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.025 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.03 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.03 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --exp-name m-aggr-kle-stop-0.05 \
    --gym-id Humanoid-v2 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo.kle.optimizations \
    --update-epochs 80 \
    --gae --norm-obs --norm-returns --norm-adv --anneal-lr --clip-vloss --weights-init orthogonal --kle-stop --target-kl 0.05 \
    --cuda \
    --prod-mode \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
