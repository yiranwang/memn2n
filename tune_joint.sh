l2=( 0.5 0.05 0.005 )

for reg in "${l2[@]}"; do
    python joint.py --regularization $reg --learning_rate 0.001 --hops 3 --embedding_size 50 --memory_size 50
done
