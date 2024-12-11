

# 1 安裝依賴
It is recommended to create a new environment with conda.
```bash
conda create -n test python=3.9
conda activate test
```
Install the required dependencies using the following commond.
```bash
pip install -r requirements.txt
```

# 2 Prepare Data
- 請下載資料集並解壓縮到專案目錄下的data資料夾中。
- 執行轉換script，將資料集轉換成embedding
```bash
python get_enmbedding.py --corpus airiti --model cohere
```
- 轉換後的取得的embedding將會存放在embd資料夾中。

# 3 降維
- 執形降維
```bash
python embed_preprocess.py \
    --embed_path embed/ecnews/embed_cohere_origin.npy \
    --embed_save_dir embed/ecnews/cohere \
    --embed_model_name cohere \
    --lang_labels_path data/ecnews/lang.txt
```


# 3 訓練模型
- 執行實驗
```bash
python main.py \
    --docs_path data/ecnews/corpus.txt \
    --labels_path data/ecnews/labels.txt \
    --lang_labels_path data/ecnews/lang.txt \
    --embeddings_path embed/ecnews/embed_cohere_origin.npy \
    --topic_word_k 15 \
    --save_dir model/ecnews/origin_768_100 \
    --seed 100 \
    --mode 'train'
```
