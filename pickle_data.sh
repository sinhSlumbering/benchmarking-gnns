size=100

cp "tsp${size}-${size}_test/tsp${size}-${size}_test.txt" data/
cp "tsp${size}-${size}_train/tsp${size}-${size}_train.txt" data/
cp "tsp${size}-${size}_val/tsp${size}-${size}_val.txt" data/

python data/TSP/TSP_Prepare.py --graph_size "${size}"

cp data/TSP.pkl data/TSP
