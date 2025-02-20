import json
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import gensim.downloader as api

# ------------------ FILE PATH ------------------
questions_path = "ADD YOUR PERSONAL QUESTION PATH HERE"
output_path = "ADD YOUR PERSONAL MODEL OUTPUT PATH HERE"
# -----------------------------------------------

def generate_raw(prompt, temperature=0.0, top_k=0, top_p=1.0, number_copy=10, gamma=5, delta=5, max_new_tokens=100):
    """
    Generates text using speculative decoding with both target and draft models.

    Args:
        prompt (str): The input text prompt.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling threshold.
        top_p (float): Top-p sampling threshold.
        number_copy (int): Number of tokens to attempt copying.
        gamma (int): Number of tokens used for similarity-based copying.
        delta (int): Number of draft tokens proposed per step.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        tuple: Generated token IDs and count of successfully copied tokens.
    """

def read_jsonl(file_path):
    """
    Reads a JSONL (JSON Lines) file and loads its content into a list.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: A list of JSON objects, each representing a line from the file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_corpus(questions, responses):
    """
    Extracts a corpus of combined question and response texts.

    Args:
        questions (list of dict): List of question objects containing question IDs and conversation turns.
        responses (list of dict): List of response objects containing corresponding replies.

    Returns:
        list of list of str: A tokenized corpus where each element is a list of words from a full conversation.
    """
    corpus = []
    for question in questions:
        question_id = question["question_id"]
        question_turns = " ".join(question["turns"])
        response = next((r for r in responses if r["question_id"] == question_id), None)
        if response:
            response_turns = " ".join(response["choices"][0]["turns"])
            combined = question_turns + " " + response_turns
            corpus.append(combined.split())
    return corpus

def generate_training_samples(corpus, k):
    """
    Creates training samples where only the last k words on the left are used as context.

    Args:
        corpus (list of list of str): Tokenized text corpus where each sentence is a list of words.
        k (int): Number of preceding words to use as context.

    Returns:
        list of tuple: A list of training samples, where each sample is a tuple (context, target).
                       - context (list of str): Last k words before the target word.
                       - target (str): The current word being predicted.
    """
    samples = []
    for sentence in corpus:
        for i in range(k, len(sentence)):
            context = sentence[i - k:i]
            target = sentence[i]
            samples.append((context, target))
    return samples

def train_word2vec_for_k(corpus, pretrained_w2v, k_values):
    """
    Trains Word2Vec models for different context sizes.

    Args:
        corpus (list of list of str): Tokenized text corpus where each sentence is a list of words.
        pretrained_w2v (gensim.models.Word2Vec): Pre-trained Word2Vec model to initialize embeddings.
        k_values (iterable of int): Different k values specifying context length.

    Returns:
        dict: A dictionary mapping each k value to its corresponding trained Word2Vec model.
    """
    models = {}
    for k in k_values:
        print(f"Training word2vec with k = {k}...")
        training_samples = generate_training_samples(corpus, k)
        sentences = [" ".join(context + [target]) for context, target in training_samples]

        word2vec = Word2Vec(
            vector_size=pretrained_w2v.vector_size,
            window=k,
            min_count=1,
            workers=4
        )
        word2vec.build_vocab([sentence.split() for sentence in sentences])

        for word in word2vec.wv.index_to_key:
            if word in pretrained_w2v:
                word2vec.wv[word] = pretrained_w2v[word]
        word2vec.train([sentence.split() for sentence in sentences], total_examples=len(sentences), epochs=10)
        models[k] = word2vec
    return models

def compute_cosine_similarity(models, corpus, k_values):
    """
    Computes the average cosine similarity between context and target words for different k values.

    Args:
        models (dict): Dictionary mapping k values to trained Word2Vec models.
        corpus (list of list of str): Tokenized text corpus where each sentence is a list of words.
        k_values (iterable of int): Different k values specifying context length.

    Returns:
        list of tuple: A list of (k, avg_similarity) pairs.
                       - k (int): Context length.
                       - avg_similarity (float): Average cosine similarity for that k.
    """
    results = []
    for k in k_values:
        model = models[k]
        similarities = []
        for sentence in corpus:
            for i in range(k, len(sentence)):
                context = sentence[i - k:i]
                target = sentence[i]
                if all(word in model.wv for word in context) and target in model.wv:
                    context_vector = sum(model.wv[word] for word in context) / len(context)
                    target_vector = model.wv[target]
                    similarity = 1 - cosine(context_vector, target_vector)
                    similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        results.append((k, avg_similarity))
    return results

def plot_results(results):
    """
    Plots the cosine similarity results for different k values.

    Args:
        results (list of tuple): A list of (k, avg_similarity) pairs.
                                 - k (int): Context length.
                                 - avg_similarity (float): Average cosine similarity for that k.

    Returns:
        None
    """
    k_values, avg_similarities = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_similarities, marker='o', linestyle='-')
    plt.xlabel("k (Context Length)")
    plt.ylabel("Average Cosine Similarity")
    plt.title("Cosine Similarity vs Context Length k")
    plt.grid()
    plt.show()

pretrained_w2v = api.load("word2vec-google-news-300")

questions = read_jsonl(questions_path)
responses = read_jsonl(output_path)
corpus = extract_corpus(questions, responses)

k_values = range(2, 30) 
models = train_word2vec_for_k(corpus, pretrained_w2v, k_values)

results = compute_cosine_similarity(models, corpus, k_values)
plot_results(results)