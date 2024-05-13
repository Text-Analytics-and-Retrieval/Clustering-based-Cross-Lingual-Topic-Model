import argparse
import time

from src.clusterTM import ClusterTM

parser = argparse.ArgumentParser(description="Cluster llm Embeddings for Topic Models")
parser.add_argument(
    "--docs_path",
    type=str,
    default="../xETM/data/training_all.txt",
    help="path to docs",
)
parser.add_argument(
    "--labels_path",
    type=str,
    default="../xETM/college_label.csv",
    help="path to labels",
)
parser.add_argument("--save_dir", type=str, help="path to save model")
parser.add_argument("--cluster_model", type=str, default="kmeans", help="cluster model")
parser.add_argument("--topic_num", type=int, default=50, help="topic number")
parser.add_argument("--topic_word_k", type=int, default=15, help="topk of topic words")
parser.add_argument(
    "--embeddings_path",
    type=str,
    help="path to embeddings",
)
parser.add_argument("--embed_dim", type=str, help="embedding dimension")
parser.add_argument(
    "--topic_model_umap_model", type=str, help="topic model umap model"
)
parser.add_argument("--umap_model_dim", type=int, help="umap model dimension")
parser.add_argument("--embedding_model", type=str, help="embedding model in Bertopic")
parser.add_argument("--lang_labels_path", type=str, help="lang labels path")
parser.add_argument("--seed", type=int, help="seed of random state")
parser.add_argument("--mode", type=str, default="train", help="mode of main.py")


args = parser.parse_args()


tic = time.time()
print("===== start =====")
tm = ClusterTM(
    docs_path=args.docs_path,
    labels_path=args.labels_path,
    save_dir=args.save_dir,
    cluster_model=args.cluster_model,
    topic_word_k=args.topic_word_k,
    embeddings_path=args.embeddings_path,
    embed_dim=args.embed_dim,
    topic_model_umap_model=args.topic_model_umap_model,
    umap_model_dim=args.umap_model_dim,
    embedding_model=args.embedding_model,
    lang_labels_path=args.lang_labels_path,
    seed=args.seed,
)
print("=" * 20 + args.mode + "=" * 20)
if args.mode == "train":
    tm.train()
tm.evaluate()

print(f"Total time: {time.time() - tic}")
print("done")
