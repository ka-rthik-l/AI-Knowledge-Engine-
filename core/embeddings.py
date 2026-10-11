from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    """
    Takes a list of strings and returns their embeddings
    """
    return model.encode(texts)
