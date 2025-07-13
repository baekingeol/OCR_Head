echo "NQ"

mkdir -p dataset/raw_data/nq
cd dataset/raw_data/nq
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
gzip -d biencoder-nq-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gzip -d biencoder-nq-train.json.gz
cd ..
cd ..
cd ..

pip install gdown
echo "hotpotqa"
mkdir -p .temp/
mkdir -p dataset/raw_data/hotpotqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O dataset/raw_data/hotpotqa/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O dataset/raw_data/hotpotqa/hotpot_dev_distractor_v1.json
rm -rf .temp/


echo "DocVQA"

python
from datasets import load_dataset
load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
