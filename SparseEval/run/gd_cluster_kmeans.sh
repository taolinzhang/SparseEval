for num_anchors in 20 40 60 80 100;
do
    echo "gd_cluster_kmeans.py $1 $num_anchors"
    python3 SparseEval/methods/gd_cluster_kmeans.py $1 $num_anchors
done