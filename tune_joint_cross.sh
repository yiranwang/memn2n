L2=( 0 0.005 0.05 0.5 )
LR=( 0.01 0.05 0.001 0.005)

dir=babi/tasks_1-20_v1-2/en/

for lamb in "${L2[@]}"; do
    for alpha in "${LR[@]}"; do
        echo tuning $lamb $alpha
        python joint.py --data_dir $dir --output_file scores_alpha_$alpha.lambda_$lamb.csv --early 50 --learning_rate $alpha --regularization $lamb | tee plogs/joint_cross_alpha_${alpha}_lambda_${lamb}.txt
    done
done
