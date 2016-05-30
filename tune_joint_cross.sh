# es=( 50 75 100 )
ms=( 50 75 100 130 150 )

dir=babi/tasks_1-20_v1-2/en/

# for e in "${es[@]}"; do
for m in "${ms[@]}"; do
    echo tuning $m
    python joint.py --data_dir $dir --output_file scores_ms_${m}.csv --early 50 --learning_rate 0.001 --memory_size $m --regularization 0 |& tee plogs/joint_cross_ms_${m}.txt
done
# done
