import os
import cohere
import argparse
import time
from random import random
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer
import numpy as np
import torch

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_MODEL = "embed-multilingual-v2.0"
XLMR_MODEL = "paraphrase-xlm-r-multilingual-v1"
MBERT_MODEL = "bert-base-multilingual-cased"
BATCH_SIZE = 64

client = cohere.Client(f"{COHERE_API_KEY}")

def get_embed(texts):
    response = client.embed(texts=texts, model=COHERE_MODEL)
    embeddings = response.embeddings
    return embeddings

parser = argparse.ArgumentParser(description="Get Embedding for Corpus")

parser.add_argument(
    "--corpus",
    type=str,
    help="Corpus name",
)
parser.add_argument(
    "--model",
    type=str,
    help="Model name",
)



args = parser.parse_args()

print("=" * 20 + "start en" + "=" * 20)
with open(f"./data/{args.corpus}/corpus.txt", "r", encoding="utf-8") as f:
    corpus_en = [line.strip() for line in f.readlines()]

if args.model == "cohere":
    p = Path(f"embed/{args.corpus}/raw") # 需要嗎？
    p.mkdir(parents=True, exist_ok=True)

    for i, k in enumerate(range(0, len(corpus_en), BATCH_SIZE)):
        time.sleep(random() * 10)
        embeddings = get_embed(corpus_en[k : k + BATCH_SIZE])
        with open(p / f"embed_{i}.txt", "w", encoding="utf-8") as f:
            for embed in embeddings:
                f.write(" ".join([str(e) for e in embed]))
                f.write("\n")

        print(f"done {i}")

elif args.model == "xlmr":
    model = SentenceTransformer(XLMR_MODEL)

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(corpus_en)

    print(f"ebeddings shape: {embeddings.shape}")
    np.save(f"embed/{args.corpus}/embed_{args.model}.npy", embeddings)
    print("done!")

elif args.model == "mbert":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL)
    model = BertModel.from_pretrained(MBERT_MODEL)
    model = model.to(device)
    def encoder(text):
        if text is None:
            return None
        # text = "Replace me by any text you'd like."
        encoded_input = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        output = model(**encoded_input)
        sent_representations = output.pooler_output.cpu().detach().numpy()
        sent_representations = sent_representations.reshape(-1)
        return sent_representations
    
    ## encode
    print("encoding...")
    sent_embeddings = map(encoder, corpus_en)
    sent_embed = np.stack(list(sent_embeddings), axis=0)
    print("encoding done")
    print(f"ebeddings shape: {sent_embed.shape}")

    ## save
    np.save(f"embed/{args.corpus}/embed_{args.model}.npy", sent_embed)
    print("done!")

