import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import SVD_process, center_SVD, centering, lang_dim_remove

parser = argparse.ArgumentParser(description="Embedding Preprocess")


parser.add_argument(
    "--embed_path",
    type=str,
    help="path of original embedding",
)
parser.add_argument(
    "--embed_save_dir",
    type=str,
    default="embed",
    help="directory to save preprocessed embedding",
)
parser.add_argument(
    "--embed_model_name",
    type=str,
    default="mbert",
    help="embedding model name",
)
parser.add_argument("--lang_labels_path", type=str, help="lang labels path")

args = parser.parse_args()

Path(args.embed_save_dir).mkdir(parents=True, exist_ok=True)

## preprocess
LANG_LABELS = np.loadtxt(args.lang_labels_path, dtype=str)
LANG_INDEX, LABEL = pd.factorize(LANG_LABELS)

print(
    f"==================== start preprocess {args.embed_model_name} ==================== "
)
# centering(args.embed_path, args.embed_save_dir, args.embed_model_name, LANG_INDEX)
SVD_process(args.embed_path, args.embed_save_dir, args.embed_model_name)
EMBED_SCALE_PATH = f"{args.embed_save_dir}/embed_{args.embed_model_name}_scale.npy"
lang_dim_remove(
    EMBED_SCALE_PATH, args.embed_save_dir, args.embed_model_name, LANG_INDEX
)
# unscale lang removal
EMBED_UNSCALE_PATH = f"{args.embed_save_dir}/embed_{args.embed_model_name}_unscale.npy"
lang_dim_remove(
    EMBED_UNSCALE_PATH, args.embed_save_dir, args.embed_model_name, LANG_INDEX, suffix="unscale_langRemoval"
)

# center SVD
center_unscale, center_scale = center_SVD(np.load(args.embed_path), LANG_INDEX)
np.save(f"{args.embed_save_dir}/embed_{args.embed_model_name}_center-unscale.npy", center_unscale)
np.save(f"{args.embed_save_dir}/embed_{args.embed_model_name}_center-scale.npy", center_scale)

for file in Path(args.embed_save_dir).iterdir():
    print(file)

print("done")
