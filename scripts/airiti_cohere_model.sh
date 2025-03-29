LM=cohere
CORPUS=airiti

# cohere origin
type=origin
for SEED in 666 777 888 168 5566; do
    echo "========== start $LM $type $DIM seed $SEED =========="
    python main.py \
        --docs_path data/clean_airiti_docs.csv \
        --labels_path data/clean_airiti_labels.csv \
        --embeddings_path embed/clean_airiti_cohere_embed.npy \
        --save_dir ${CORPUS}/${LM}/${type}_${SEED} \
        --lang_labels_path data/airiti_lang_labels.csv \
        --seed $SEED \
        --mode 'train'
    echo "${LM}_${type} seed ${SEED} done."
done

# cohere umap
# type=umap
# for DIM in 100 200 500; do
#     for SEED in 666 777 888 168 5566; do
#         echo "========== start cohere $type $DIM seed $SEED =========="
#         python main.py \
#             --docs_path data/clean_airiti_docs.csv \
#             --labels_path data/clean_airiti_labels.csv \
#             --embeddings_path embed/clean_airiti_cohere_embed.npy \
#             --topic_model_umap_model $type \
#             --umap_model_dim $DIM \
#             --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
#             --lang_labels_path data/airiti_lang_labels.csv \
#             --seed $SEED \
#             --mode 'train'
#         echo "${LM}_${type} $DIM seed $SEED done."
#     done
# done

# cohere tsne
# type=tsne
# for DIM in 100 200 500; do
#     for SEED in 666 777 888 168 5566; do
#         echo "========== start cohere $type $DIM seed $SEED =========="
#         python main.py \
#             --docs_path data/clean_airiti_docs.csv \
#             --labels_path data/clean_airiti_labels.csv \
#             --embeddings_path embed/clean_airiti_cohere_embed.npy \
#             --topic_model_umap_model $type \
#             --umap_model_dim $DIM \
#             --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
#             --lang_labels_path data/airiti_lang_labels.csv \
#             --seed $SEED \
#             --mode 'train'
#         echo "${LM}_${type} $DIM seed $SEED done."
#     done
# done

# cohere unscale
type=unscale
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start cohere $type $DIM seed $SEED =========="
        python main.py \
            --docs_path data/clean_airiti_docs.csv \
            --labels_path data/clean_airiti_labels.csv \
            --embeddings_path embed/cohere/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
            --lang_labels_path data/airiti_lang_labels.csv \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done

# cohere scale
type=scale
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start cohere $type $DIM seed $SEED =========="
        python main.py \
            --docs_path data/clean_airiti_docs.csv \
            --labels_path data/clean_airiti_labels.csv \
            --embeddings_path embed/cohere/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
            --lang_labels_path data/airiti_lang_labels.csv \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done

# # cohere center
# type=center
# for DIM in 768; do
#     for SEED in 666 777 888 168 5566; do
#         echo "========== start cohere $type $DIM seed $SEED =========="
#         python main.py \
#             --docs_path data/clean_airiti_docs.csv \
#             --labels_path data/clean_airiti_labels.csv \
#             --embeddings_path embed/cohere/embed_${LM}_${type}.npy \
#             --embed_dim $DIM \
#             --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
#             --lang_labels_path data/airiti_lang_labels.csv \
#             --seed $SEED \
#             --mode 'train'
#         echo "${LM}_${type} $DIM seed $SEED done."
#     done
# done

# cohere langRemoval
type=langRemoval
for DIM in 100 200 500; do
    for SEED in 666 777 888 168 5566; do
        echo "========== start cohere $type $DIM seed $SEED =========="
        python main.py \
            --docs_path data/clean_airiti_docs.csv \
            --labels_path data/clean_airiti_labels.csv \
            --embeddings_path embed/cohere/embed_${LM}_${type}.npy \
            --embed_dim $DIM \
            --save_dir ${CORPUS}/${LM}/${type}_${DIM}_${SEED} \
            --lang_labels_path data/airiti_lang_labels.csv \
            --seed $SEED \
            --mode 'train'
        echo "${LM}_${type} $DIM seed $SEED done."
    done
done
