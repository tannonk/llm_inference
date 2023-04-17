from typing import List, Tuple, Optional
import logging
import pickle
import argparse
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

from utils.helpers import iter_text_lines

logger = logging.getLogger(__name__)

model = SentenceTransformer('all-mpnet-base-v2')

def save_embedded_sents(sentences: List[str], embeddings: np.array, embed_path: str) -> None:
    """
    Save sentences and their embeddings to a file.
    """
    logger.info(f"Saving embeddings to {embed_path}")
    with open(embed_path, "wb") as outf:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, outf, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_embedded_sents(embed_path: str) -> Tuple[List[str], np.ndarray]:
    with open(embed_path, 'rb') as f:
        data = pickle.load(f)
    return data['sentences'], data['embeddings']

def encode_sentences(sentences: List[str], show_progress_bar: bool = True, device: str = 'cuda', embed_path: Optional[str] = None):
    """
    Encode sentences using the sentence transformer model and save the embeddings to a file for later use.
    """
    embeddings = model.encode(sentences, show_progress_bar=True, device='cuda', convert_to_numpy=True)

    if embed_path:
        save_embedded_sents(sentences, embeddings, embed_path=embed_path)
        
    return embeddings

def fetch_embeddings(embed_path: str, sentences: List[str]) -> np.ndarray:
    """
    Fetch embeddings from a saved file if they exist, otherwise compute them and save them to a file.
    """
    try:
        saved_sentences, saved_embeddings = load_embedded_sents(embed_path)
        if saved_sentences == sentences:
            return saved_embeddings
        else:
            logger.info("Sentences have changed. Computing new embeddings.")
            embeddings = encode_sentences(sentences, embed_path=embed_path)
            return embeddings
    except FileNotFoundError:
        logger.info("Embeddings not found. Computing new embeddings.")
        embeddings = encode_sentences(sentences, embed_path=embed_path)
        return embeddings

def fetch_most_similar(query: str, sentences: List[str], embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[int, str, float]]:
    """
    Fetch the most similar sentences from embedding matrix to a query sentence.
    """

    query_embedding = model.encode(query, show_progress_bar=False, device='cuda', convert_to_numpy=True)
    cosine_scores = pytorch_cos_sim(query_embedding, embeddings)

    #Sort scores in decreasing order
    scores = cosine_scores[0].tolist()
    sorted_scores = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_scores = sorted_scores[:top_k]

    return [(i, sentences[i], scores[i]) for i in sorted_scores]


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", type=str, required=False, help="Path to input file")
    ap.add_argument("-e", "--embed_path", type=str, required=False, help="Path to save/load embeddings")
    ap.add_argument("-q", "--query", type=str, required=False, help="Query sentence")
    ap.add_argument("-k", "--top_k", type=int, required=False, help="Number of most similar sentences to return")
    return ap.parse_args()

if __name__ == "__main__":

    args = set_args()

    #Load sentences
    sentences = list(iter_text_lines(args.input_file))

    # Split sentences into source and target, keeping only source (v0 from Newsela)
    sentences = [s.split('\t')[0] for s in sentences]
    
    #Compute embeddings
    embeddings = fetch_embeddings(args.embed_path, sentences)

    similar_sentences = fetch_most_similar(args.query, sentences, embeddings, top_k=args.top_k)
    print(similar_sentences)
