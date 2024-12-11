# Clustering-based-Cross-Lingual-Topic-Model
This is the implementation of the paper "Refining Dimensions for Improving Clustering-based Cross-lingual Topic Models". This repository contains the code for the u-SVD, SVD-LR and the evaluation  on three datasets: Airiti, ECNews, and Rakuten.

## 1 Install
It is recommended to create a new environment with conda.
```bash
conda create -n test python=3.9
conda activate test
```
Install the required dependencies using the following commond.
```bash
pip install -r requirements.txt
conda install -c conda-forge multicore-tsne
```

## 2 Prepare Dataset
Please download the data from the following link and put it in the `data` folder. [download](https://drive.google.com/file/d/1vHYhrfeTDATZWXvHxPtw1oW3Ii56jse6/view?usp=sharing)
default data structure is as follows:
```
data
├── airiti
│   ├── corpus.txt
│   ├── labels.txt
│   └── lang.txt
├── ecnews
│   ├── corpus.txt
│   ├── labels.txt
│   └── lang.txt
└── rakuten
    ├── corpus.txt
    ├── labels.txt
    └── lang.txt
```
If you want to run evaluation on airiti dataset, you need to download the reference corpus from the following link and put it in the `eval-data` folder. [download](https://drive.google.com/file/d/1pe0EQ2qgilwnOFai9b0cRCemOJBnpr1F/view?usp=sharing)

## 3 Get Embedding
You can run the following command to get the embedding of the corpus, Or you can download the embedding which we have already generated from the following link and put it in the `embed` folder. [download](https://drive.google.com/file/d/1x-q2PcnEqjAr8J_p0mIeih1vxlBGGQ2z/view?usp=sharing)
```bash
python get_enmbedding.py --corpus ecnews --model cohere
```

## 4 Build dimansion reduction embedding
Use `embed_preprocess.py` to build dimansion-reduction embedding. To run the full experiment, ypu can check the script in `scripts/embed_preprocess.sh`.
```bash
python embed_preprocess.py \
    --embed_path embed/ecnews/embed_cohere_origin.npy \
    --embed_save_dir embed/ecnews/cohere \
    --embed_model_name cohere \
    --lang_labels_path data/ecnews/lang.txt
```
Here is the example result by running the above command:
```
├── embed
│   ├── ecnews
│   │   ├── cohere
│   │   │   ├── embed_cohere_center.npy
│   │   │   ├── embed_cohere_center-scale.npy
│   │   │   ├── embed_cohere_center-unscale.npy
│   │   │   ├── embed_cohere_langRemoval.npy 
│   │   │   ├── embed_cohere_scale.npy
│   │   │   ├── embed_cohere_unscale_langRemoval.npy # SVD-LR
│   │   │   └── embed_cohere_unscale.npy # u-SVD
```

## 5 Run Experiment
Use `main.py` to run the experiment. To run the full experiment, you can check the script in `scripts/ecnews_cohere_model.sh`. For a dataset, we provide three type of embedding methods: `cohere`, `mbert`, and `xlmr`.
```bash
python main.py \
    --docs_path data/ecnews/corpus.txt \
    --labels_path data/ecnews/labels.txt \
    --lang_labels_path data/ecnews/lang.txt \
    --embeddings_path embed/ecnews/cohere/embed_cohere_unscale.npy \
    --topic_word_k 15 \
    --embed_dim 100 \
    --save_dir ecnews/unscale_100_100 \
    --seed 666 \
    --mode 'train'
```


## 📚 Citation
If you find our work useful, please consider citing our work:
```
TBD
```
