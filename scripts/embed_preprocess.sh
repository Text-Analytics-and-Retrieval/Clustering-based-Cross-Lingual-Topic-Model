# airiti
corpus=airiti
python embed_preprocess.py \
    --embed_path embed/${corpus}/embed_cohere_origin.npy \
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

## rakuten
corpus=rakuten
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
