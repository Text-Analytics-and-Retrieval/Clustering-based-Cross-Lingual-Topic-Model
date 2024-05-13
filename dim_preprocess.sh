# airiti
corpus=airiti
python embed_preprocess.py \
    --embed_path embed/clean_airiti_cohere_embed.npy \
    --embed_save_dir embed/${corpus}/cohere \
    --embed_model_name cohere \
    --lang_labels_path data/${corpus}/airiti_lang_labels.csv

python embed_preprocess.py \
    --embed_path embed/embed_bert-base-multilingual-cased.npy \
    --embed_save_dir embed/${corpus}/mbert \
    --embed_model_name mbert \
    --lang_labels_path data/${corpus}/airiti_lang_labels.csv

python embed_preprocess.py \
    --embed_path embed/clean_airiti_sbert_embed.npy \
    --embed_save_dir embed/${corpus}/xlmr \
    --embed_model_name xlmr \
    --lang_labels_path data/${corpus}/airiti_lang_labels.csv

## rakuten
corpus=rakuten
python embed_preprocess.py \
    --embed_path embed/${corpus}/cohere/embed_cohere_origin.npy \
    --embed_save_dir embed/${corpus}/cohere \
    --embed_model_name cohere \
    --lang_labels_path data/${corpus}/lang.txt | tee rakuten_embed_preprocess.log

python embed_preprocess.py \
    --embed_path embed/${corpus}/embed_bert-base-multilingual-cased.npy \
    --embed_save_dir embed/${corpus}/mbert \
    --embed_model_name mbert \
    --lang_labels_path data/${corpus}/lang.txt | tee -a rakuten_embed_preprocess.log

python embed_preprocess.py \
    --embed_path embed/${corpus}/embed_paraphrase-xlm-r-multilingual-v1.npy \
    --embed_save_dir embed/${corpus}/xlmr \
    --embed_model_name xlmr \
    --lang_labels_path data/${corpus}/lang.txt | tee -a rakuten_embed_preprocess.log

# ecnews
corpus=ecnews
python embed_preprocess.py \
    --embed_path embed/${corpus}/cohere/embed_cohere_origin.npy \
    --embed_save_dir embed/${corpus}/cohere \
    --embed_model_name cohere \
    --lang_labels_path data/${corpus}/lang.txt

python embed_preprocess.py \
    --embed_path embed/${corpus}/embed_bert-base-multilingual-cased.npy \
    --embed_save_dir embed/${corpus}/mbert \
    --embed_model_name mbert \
    --lang_labels_path data/${corpus}/lang.txt

python embed_preprocess.py \
    --embed_path embed/${corpus}/embed_paraphrase-xlm-r-multilingual-v1.npy \
    --embed_save_dir embed/${corpus}/xlmr \
    --embed_model_name xlmr \
    --lang_labels_path data/${corpus}/lang.txt
