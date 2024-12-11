LM=mbert
CORPUS=airiti

# m-bert only
type=origin
DIM=768
for SEED in 666 777 888 168 5566; do
    echo "========== start $LM $type $DIM seed $SEED =========="
    python main.py \
        --docs_path data/clean_airiti_docs.csv \
        --labels_path data/clean_airiti_labels.csv \
        --embeddings_path embed/embed_bert-base-multilingual-cased.npy \
        --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
        --lang_labels_path data/airiti_lang_labels.csv \
        --seed $SEED \
        --mode 'train'
    echo "${LM}_${type} seed ${SEED} done."
done

# mbert + UMAP
type=umap
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path data/clean_airiti_docs.csv \
            --labels_path data/clean_airiti_labels.csv \
            --embeddings_path embed/embed_bert-base-multilingual-cased.npy \
            --topic_model_umap_model $type \
            --umap_model_dim $DIM \
            --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
            --lang_labels_path data/airiti_lang_labels.csv \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} seed ${SEED} done."
    done
done

# mbert + tsne
type=tsne
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path data/clean_airiti_docs.csv \
            --labels_path data/clean_airiti_labels.csv \
            --embeddings_path embed/embed_bert-base-multilingual-cased.npy \
            --topic_model_umap_model $type \
            --umap_model_dim $DIM \
            --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
            --lang_labels_path data/airiti_lang_labels.csv \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} seed ${SEED} done."
    done
done

# m-bert center
type=center
for DIM in 768; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path data/clean_airiti_docs.csv \
            --labels_path data/clean_airiti_labels.csv \
            --embeddings_path embed/${LM}/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
            --lang_labels_path data/airiti_lang_labels.csv \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done

# m-bert unscale
type=unscale
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path data/clean_airiti_docs.csv \
            --labels_path data/clean_airiti_labels.csv \
            --embeddings_path embed/${LM}/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
            --lang_labels_path data/airiti_lang_labels.csv \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done

# m-bert scale
type=scale
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path data/clean_airiti_docs.csv \
            --labels_path data/clean_airiti_labels.csv \
            --embeddings_path embed/${LM}/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
            --lang_labels_path data/airiti_lang_labels.csv \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done

# m-bert langRemoval
LM=mbert
CORPUS=airiti
type=langRemoval
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path data/clean_airiti_docs.csv \
            --labels_path data/clean_airiti_labels.csv \
            --embeddings_path embed/${LM}/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
            --lang_labels_path data/airiti_lang_labels.csv \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done
