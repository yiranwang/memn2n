L2=( 0 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 )

dir=../cs224d-project/data/babi/tasks_1-20_v1-2/en/

for lamb in "${L2[@]}"; do
    echo tuning $lamb
    python joint.py --data_dir $dir --early 50 --learning_rate 0.001 --regularization $lamb > joint_alpha_0.001_lambda_$lamb.txt
done
