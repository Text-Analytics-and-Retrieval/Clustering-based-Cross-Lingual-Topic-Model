import json
import pickle
import subprocess
import time
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy.spatial import distance
from sklearn.cluster import KMeans
from typing_extensions import Literal
from umap import UMAP

from src.evaluator import (compute_purity, divetsity,
                           documents_to_cooccurence_matrix, npmi_score,
                           split_topk_topic_word)
from src.model import GMM_MODEL, TSNE


class ClusterTM:
    def __init__(
        self,
        docs_path: str,
        labels_path: str,
        save_dir: str,
        cluster_model: Literal["kmeans"] = "kmeans",
        topic_num: int = 50,
        topic_word_k: int = 15,
        embeddings_path: Optional[str] = None,
        embed_dim: Optional[int] = None,
        topic_model_umap_model: Optional[Literal["umap", "tsne"]] = None,
        umap_model_dim: Optional[int] = None,
        lang_labels_path: Optional[str] = None,
        seed: int = 1,
        **kwargs,
    ):
        self.docs_path = docs_path
        self.labels_path = labels_path
        self.cluster_model = cluster_model
        self.topic_num = topic_num
        self.topic_word_k = topic_word_k
        self.embeddings_path = embeddings_path
        self.evaluate_value: dict = dict()
        self.topic_model_umap_model = topic_model_umap_model
        self.umap_model_dim = umap_model_dim
        self.seed = seed
        self.kwargs = kwargs

        ## data preprocessing
        # load docs and labels
        docs = pd.read_csv(self.docs_path, header=None, sep="\t", names=["text"])
        labels = pd.read_csv(self.labels_path, header=None, names=["label"])
        assert len(docs) == len(labels)

        # load SVD embedding
        if embeddings_path:
            embeddings = np.load(self.embeddings_path)
            assert len(docs) == len(embeddings)
        else:
            embeddings = None

        # TODO: 要設計這個變數從哪裡傳入
        if not embed_dim:
            embed_dim = None
        else:
            embed_dim = int(embed_dim)
            embeddings = embeddings[:, :embed_dim]
            assert len(docs) == len(embeddings)

        ## lang labels
        lang_labels = np.loadtxt(lang_labels_path, dtype=str)
        assert len(docs) == len(lang_labels)
        self.lang_labels = lang_labels
        self.lang_index, _ = pd.factorize(lang_labels)

        docs_lang1 = docs.iloc[self.lang_index == 0]
        docs_lang2 = docs.iloc[self.lang_index == 1]
        assert docs.shape[0] == labels.shape[0]

        if self.embeddings_path:
            assert docs.shape[0] == labels.shape[0] == embeddings.shape[0]
            print(f"embedding dim: {embeddings.shape}")

        self.docs = docs["text"].tolist()
        self.docs_lang1 = docs_lang1["text"].map(lambda x: x.split(" ")).tolist()
        self.docs_lang2 = docs_lang2["text"].map(lambda x: x.split(" ")).tolist()
        self.labels = labels
        self.embeddings = embeddings

        ## saving path
        self.save_dir = Path(f"model/{save_dir}")
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

    def train(self):
        # training topic model
        if self.cluster_model == "kmeans":
            cluster_model = KMeans(n_clusters=self.topic_num, random_state=self.seed)
        self.umap_model = BaseDimensionalityReduction()

        if self.topic_model_umap_model == "umap":
            self.umap_model = (
                UMAP(
                    n_components=int(self.umap_model_dim),
                    random_state=self.seed,
                )
                if self.umap_model_dim
                else None
            )

        if self.topic_model_umap_model == "tsne":
            self.umap_model = (
                TSNE(
                    n_components=int(self.umap_model_dim), 
                    random_state=self.seed
                )
                if self.umap_model_dim
                else TSNE(n_components=5)
            )

        # start fitting
        print("=" * 20 + str(cluster_model) + "=" * 20)
        print(f"embedding dim: {self.embeddings.shape}")
        tic = time.time()
        self.topic_model = BERTopic(
            language=None,
            umap_model=self.umap_model,
            hdbscan_model=cluster_model,
            verbose=True,
            **self.kwargs,
        )
        topics, probs = self.topic_model.fit_transform(self.docs, self.embeddings)
        print(f"dim Reduction model: {self.topic_model.umap_model}")
        del self.topic_model.umap_model
        print(f"Time: {time.time() - tic:.2f} sec")

        self.topic_model.save(
            str(self.save_dir / "topic-model.pkl"), save_embedding_model=False
        )

        print(self.topic_model.get_topic_info())

        all_topics = self.topic_model.get_topics()
        pprint(all_topics[0][:10])
        # pprint(all_topics[1][:10])

        topic_word = dict()
        for i in all_topics:
            topic_word[i] = list(map(lambda x: x[0], all_topics[i]))[:20]

        with (self.save_dir / "topic_words_phi.json").open("w") as f:
            json.dump(topic_word, f, ensure_ascii=False, indent=4)

        # save thetas
        topic_distr, _ = self.topic_model.approximate_distribution(self.docs)
        np.save(self.save_dir / "topic_distr.npy", topic_distr)

    def cnpmi_wiki(self, lang1_path: Path, lang2_path: Path, lang2:str="zh"):
        # 定義命令
        command = [
            "python",
            "CNPMI.py",
            "--topics1",
            f"{str(lang1_path)}",
            "--topics2",
            f"{str(lang2_path)}",
            "--ref_corpus_config",
            f"CNPMI/configs/ref_corpus/en_{lang2}.yaml",
        ]

        # 執行命令，並捕獲輸出
        try:
            completed_process = subprocess.run(
                command,
                check=True,
                shell=False,
                cwd="CNPMI",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # 獲取標準輸出和標準錯誤
            stdout = completed_process.stdout
            stderr = completed_process.stderr

            if stdout:
                print(stdout)
                lines = stdout.strip().splitlines()
                self.evaluate_value["c-npmi"] = lines[-1]

            if stderr:
                print("標準錯誤：")
                print(stderr)

            print("cnpmi repo success.")
        except subprocess.CalledProcessError as e:
            print(f"執行 bash 命令時出錯：{e}")

    def cnpmi_comparable(
        self, lang1_topic: List[List[str]], lang2_topic: List[List[str]], topk: int
    ):
        eval_data_path = (
            Path("eval_data") / f"{str(Path(self.docs_path).parent).replace('/', '-')}"
        )
        eval_data_path.mkdir(exist_ok=True, parents=True)
        if not (eval_data_path / "cnpmi_eval_data.pkl").exists():
            (
                cooccurence_matrix,
                _,
                compound_dictionary,
                num_of_documents,
            ) = documents_to_cooccurence_matrix(
                source_language_documents=self.docs_lang1,
                target_language_documents=self.docs_lang2,
            )
            with (eval_data_path / "cnpmi_eval_data.pkl").open("wb") as f:
                pickle.dump(
                    (
                        cooccurence_matrix,
                        compound_dictionary,
                        num_of_documents,
                    ),
                    f,
                    protocol=4,
                )
        with (eval_data_path / "cnpmi_eval_data.pkl").open("rb") as f:
            (
                cooccurence_matrix,
                compound_dictionary,
                num_of_documents,
            ) = pickle.load(f)
        self.evaluate_value["c-npmi"] = npmi_score(
            cn_topic=lang1_topic,
            en_topic=lang2_topic,
            topk=topk,
            cooccurence_matrix=cooccurence_matrix,
            compound_dictionary=compound_dictionary,
            num_of_documents=num_of_documents,
            coherence_method="npmi",
        )

    def evaluate(self):
        ## load model
        self.topic_model = BERTopic.load(str(self.save_dir / "topic-model.pkl"))
        self.topic_distr = np.load(self.save_dir / "topic_distr.npy")

        ## labels ratio in each topic
        self.topic_label = self.topic_model.topics_
        topic_lang_ratio = pd.DataFrame(
            {
                "topic_label": self.topic_label,
                "lang_label": self.lang_labels,
            }
        )
        inverse_jsd_list = []
        groups = topic_lang_ratio.groupby(["topic_label"])
        for g, table in groups:
            lang_ratio = table["lang_label"].value_counts(normalize=True)
            value = 1 - distance.jensenshannon(lang_ratio.values, np.array([0.5, 0.5]))
            inverse_jsd_list.append(value)
        self.evaluate_value["inverse_jsd"] = np.mean(inverse_jsd_list)
        print(f"mean inverse_jsd: {self.evaluate_value.get('inverse_jsd'):.4f}")

        ## purity
        assert len(self.topic_model.topics_) == len(self.labels)
        purity, purity_results = compute_purity(
            pd.DataFrame(
                {
                    "topic_labels": self.topic_model.topics_,
                    "college_labels": self.labels["label"],
                }
            )
        )
        self.evaluate_value["purity"] = purity
        print(f"purity : {self.evaluate_value.get('purity'):.4f}")

        self.purity_results = purity_results

        ## phi metric
        # count topk chinese words and english words
        topic_word_matrix = self.topic_model.c_tf_idf_.toarray()
        feature = self.topic_model.vectorizer_model.get_feature_names()
        en_topic, cn_topic = split_topk_topic_word(
            topic_word_matrix=topic_word_matrix, feature=feature, topk=self.topic_word_k
        )

        # diversity
        self.evaluate_value["diversity"] = (
            divetsity(self.topic_num, en_topic) + divetsity(self.topic_num, cn_topic)
        ) / 2
        print(f"diversity: {self.evaluate_value.get('diversity')}")

        # save topk topic words
        tpword_en_p = self.save_dir / f"top{self.topic_word_k}_topic_word_en.txt"
        tpword_cn_p = self.save_dir / f"top{self.topic_word_k}_topic_word_cn.txt"
        with tpword_en_p.open("w") as f:
            for en in en_topic:
                f.write(" ".join(en) + "\n")
        with tpword_cn_p.open("w") as f:
            for cn in cn_topic:
                f.write(" ".join(cn) + "\n")

        # # compute c-npmi
        # indomain
        if "airiti" in self.docs_path:
            self.cnpmi_comparable(en_topic, cn_topic, topk=self.topic_word_k)
        if 'ecnews' in self.docs_path:
            self.cnpmi_wiki(tpword_en_p, tpword_cn_p)
        if 'rakuten' in self.docs_path:
            self.cnpmi_wiki(tpword_en_p, tpword_cn_p, lang2="ja")
        print(f"c-npmi score: {self.evaluate_value.get('c-npmi')}")

        ## save metrics
        with (self.save_dir / "evaluate.json").open("w") as f:
            json.dump(self.evaluate_value, f, ensure_ascii=False, indent=4)

    def _back_reflexibility(self):
        """這是 archived reflexibility 的計算方式"""
        ## reflexibility
        true_distribution = (
            self.labels["label"]
            .value_counts(normalize=True)
            .to_frame(name="true_proportion")
        )
        margin_topic_distri = np.nansum(self.topic_distr, axis=0) / np.nansum(
            self.topic_distr
        )
        self.purity_results["margin"] = margin_topic_distri
        topic_distri = self.purity_results.groupby("topic_label")["margin"].sum()
        self.topic_distri = topic_distri

        mapper_list = self.purity_results.loc[:, ["topic", "topic_label"]].to_dict(
            orient="records"
        )
        mapper = {pair["topic"]: pair["topic_label"] for pair in mapper_list}
        topic_distribution = (
            pd.Series(map(mapper.get, self.topic_model.topics_))
            .value_counts(normalize=True)
            .to_frame("topic_distri")
        )
        true_distribution = true_distribution.join(topic_distribution)
        true_distribution = true_distribution.join(self.topic_distri)
        true_distribution.fillna(0, inplace=True)
        reflexibility = distance.jensenshannon(
            true_distribution["true_proportion"], true_distribution["topic_distri"]
        )
        theta_reflexibility = distance.jensenshannon(
            true_distribution["true_proportion"], true_distribution["margin"]
        )

        self.evaluate_value["reflexibility"] = reflexibility
        self.evaluate_value["theta_reflexibility"] = theta_reflexibility
        # print(f"the reflexibility (JSD) of model to true distribution: {reflexibility}")
        print(
            f"the theta reflexibility (JSD) of model to true distribution: {theta_reflexibility}"
        )
