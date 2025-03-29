LM=cohere
CORPUS=rakuten

DOCS_PATH=data/${CORPUS}/corpus.txt
LABELS_PATH=data/${CORPUS}/labels.txt
EMBED_DIR=embed/${CORPUS}/${LM}
LANG_LABELS_PATH=data/${CORPUS}/lang.txt

SAVE_DIR=${CORPUS}/${LM}

# cohere origin
type=origin
DIM=768
for SEED in 666 777 888 168 5566; do
    echo "========== start $LM $type $DIM seed $SEED =========="
    python main.py \
        --docs_path ${DOCS_PATH} \
        --labels_path ${LABELS_PATH} \
        --embeddings_path ${EMBED_DIR}/embed_cohere_origin.npy \
        --save_dir ${SAVE_DIR}/${type}_${DIM}_${SEED} \
        --lang_labels_path ${LANG_LABELS_PATH} \
        --seed $SEED \
        --mode 'train'
    echo "${LM}_${type} seed ${SEED} done."
done

# cohere + UMAP
# type=umap
# for DIM in 100 200 500; do
#     for SEED in 666 777 888 168 5566; do
#         echo "========== start $LM $type $DIM seed $SEED =========="
#         python main.py \
#             --docs_path ${DOCS_PATH} \
#             --labels_path ${LABELS_PATH} \
#             --embeddings_path ${EMBED_DIR}/embed_cohere_origin.npy \
#             --topic_model_umap_model $type \
#             --umap_model_dim $DIM \
#             --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
#             --lang_labels_path ${LANG_LABELS_PATH} \
#             --seed $SEED \
#             --mode 'train'
#         echo "${LM}_${type} seed ${SEED} done."
#     done
# done

# cohere tsne
# type=tsne
# for DIM in 100 200 500; do
#     for SEED in 666 777 888 168 5566; do
#         echo "========== start $LM $type $DIM seed $SEED =========="
#         python main.py \
#             --docs_path ${DOCS_PATH} \
#             --labels_path ${LABELS_PATH} \
#             --embeddings_path ${EMBED_DIR}/embed_cohere_origin.npy \
#             --topic_model_umap_model $type \
#             --umap_model_dim $DIM \
#             --save_dir ${SAVE_DIR}/${type}_${DIM}_${SEED} \
#             --lang_labels_path ${LANG_LABELS_PATH} \
#             --seed $SEED \
#             --mode 'train'
#         echo "${LM}_${type} seed ${SEED} done."
#     done
# done

# cohere center
# type=center
# for DIM in 768; do
#     for SEED in 666 777 888 168 5566; do
#         echo "========== start $LM $type $DIM seed $SEED =========="
#         python main.py \
#             --docs_path ${DOCS_PATH} \
#             --labels_path ${LABELS_PATH} \
#             --embeddings_path ${EMBED_DIR}/embed_${LM}_${type}.npy \
#             --embed_dim $DIM \
#             --save_dir ${SAVE_DIR}/${type}_${DIM}_${SEED} \
#             --lang_labels_path ${LANG_LABELS_PATH} \
#             --seed $SEED \
#             --mode 'train'
#         echo "${LM}_${type} $DIM seed $SEED done."
#     done
# done

# cohere unscale
type=unscale
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path ${DOCS_PATH} \
            --labels_path ${LABELS_PATH} \
            --embeddings_path ${EMBED_DIR}/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${SAVE_DIR}/${type}_${DIM}_${SEED} \
            --lang_labels_path ${LANG_LABELS_PATH} \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done

# cohere scale
type=scale
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path ${DOCS_PATH} \
            --labels_path ${LABELS_PATH} \
            --embeddings_path ${EMBED_DIR}/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${SAVE_DIR}/${type}_${DIM}_${SEED} \
            --lang_labels_path ${LANG_LABELS_PATH} \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done

# cohere langRemoval
type=langRemoval
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path ${DOCS_PATH} \
            --labels_path ${LABELS_PATH} \
            --embeddings_path ${EMBED_DIR}/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${SAVE_DIR}/${type}_${DIM}_${SEED} \
            --lang_labels_path ${LANG_LABELS_PATH} \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done

# unscale_langRemoval
type=unscale_langRemoval
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start $LM $type $DIM seed $SEED =========="
        python main.py \
            --docs_path ${DOCS_PATH} \
            --labels_path ${LABELS_PATH} \
            --embeddings_path ${EMBED_DIR}/embed_${LM}_${type}.npy \
            --topic_word_k 15 \
            --embed_dim $DIM \
            --save_dir ${SAVE_DIR}/${type}_${DIM}_${SEED} \
            --lang_labels_path ${LANG_LABELS_PATH} \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done
