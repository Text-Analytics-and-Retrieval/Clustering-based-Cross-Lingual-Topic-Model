import dask.array as da
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def embed_centering(embeddings, lang_index):
    embeddings_1 = embeddings[lang_index == 0]
    embeddings_2 = embeddings[lang_index == 1]

    embeddings_1_center = embeddings_1 - embeddings_1.mean(axis=0)
    embeddings_2_center = embeddings_2 - embeddings_2.mean(axis=0)
    embeddings_center = np.concatenate(
        (embeddings_1_center, embeddings_2_center), axis=0
    )  # 2d array    
    return embeddings_center

def embed_SVD(embeddings):
    embeddings = da.from_array(embeddings)
    u, s, v = da.linalg.svd(embeddings)
    print(f"SVD shape: {u.shape, s.shape, v.shape}")
    return u.compute(), (u * s).compute()

def centering(embed_path, embed_savedir, embed_model_name, lang_index):
    embeddings = np.load(embed_path)
    embeddings_1 = embeddings[lang_index == 0]
    embeddings_2 = embeddings[lang_index == 1]

    embeddings_1_center = embeddings_1 - embeddings_1.mean(axis=0)
    embeddings_2_center = embeddings_2 - embeddings_2.mean(axis=0)
    embeddings_center = np.concatenate(
        (embeddings_1_center, embeddings_2_center), axis=0
    )  # 2d array
    np.save(f"{embed_savedir}/embed_{embed_model_name}_center.npy", embeddings_center)

    return None

## SVD process
def SVD_process(embed_path, embed_savedir, embed_model_name):
    embeddings = np.load(embed_path)
    embeddings = da.from_array(embeddings)

    u, s, v = da.linalg.svd(embeddings)

    u.shape, s.shape, v.shape

    u_weighted = u * s

    embed_svd_unscale = u.compute()
    embed_svd_unscale.shape

    np.save(f"{embed_savedir}/embed_{embed_model_name}_unscale.npy", embed_svd_unscale)

    embed_svd_scale = u_weighted.compute()

    np.save(
        f"{embed_savedir}/embed_{embed_model_name}_scale.npy",
        embed_svd_scale,
    )

    return None


## language dimension removal
def lang_dim_remove(embed_scale_path, embed_savedir, embed_model_name, lang_index, suffix="langRemoval"):
    embed_scale = np.load(embed_scale_path)

    embed_scale_lang1 = embed_scale[lang_index == 0]
    embed_scale_lang2 = embed_scale[lang_index == 1]

    def t_test(dim):
        res = ttest_ind(embed_scale_lang1[:, dim], embed_scale_lang2[:, dim])
        return res.statistic, res.pvalue

    res = list(map(t_test, range(embed_scale_lang1.shape[1])))
    ttest_df = pd.DataFrame(res, columns=["statistic", "pvalue"])
    ttest_df["statistic"] = ttest_df["statistic"].abs()
    max_statistic_idx = ttest_df["statistic"].idxmax()
    embed_scale_lang_removal = np.delete(embed_scale, max_statistic_idx, axis=1)
    np.save(
        f"{embed_savedir}/embed_{embed_model_name}_{suffix}.npy",
        embed_scale_lang_removal,
    )

    return None

def center_SVD(embeddings, lang_index):
    embeddings_center = embed_centering(embeddings, lang_index)
    unscale, scale = embed_SVD(embeddings_center)
    return unscale, scale


if __name__ == "__main__":
    embed_path = "embed/embed_bert-base-multilingual-cased.npy"
    SVD_process(embed_path, "embed", "mbert")
    embed_path = "embed/embed_mbert_scale.npy"
    lang_dim_remove(embed_path, "embed", "mbert")
    print("done")
