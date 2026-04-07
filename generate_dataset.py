import json
import os
import random

from openai import OpenAI

DOMAIN = "Machine Learning and Artificial Intelligence"
N_SAMPLES = 60
TRAIN_RATIO = 0.9
OUTPUT_DIR = "data"
RANDOM_SEED = 42

TOPICS = [
    "linear regression",
    "logistic regression",
    "decision trees",
    "random forests",
    "gradient boosting",
    "artificial neural networks",
    "convolutional neural networks (CNN)",
    "recurrent neural networks (RNN/LSTM)",
    "transformers and attention mechanism",
    "reinforcement learning",
    "clustering (K-Means, DBSCAN)",
    "dimensionality reduction (PCA, t-SNE)",
    "model evaluation metrics",
    "overfitting and regularization",
    "transfer learning",
    "LLM fine-tuning",
    "model quantization (QLoRA/LoRA)",
    "feature engineering",
    "ML pipelines and MLOps",
    "ethics and bias in AI",
]

SYSTEM_PROMPT = (
    "You are an expert professor in Machine Learning and Artificial Intelligence. "
    "Generate didactic, clear and technically accurate instruction pairs in English. "
    "Reply ONLY with a JSON object in the format: "
    '{"prompt": "<question>", "response": "<full answer>"}'
)


def generate_pair(client: OpenAI, topic: str) -> dict:
    user_msg = (
        f"Create a question and detailed answer pair about the topic: '{topic}'. "
        "The question should be the type a student would ask a professor. "
        "The answer should have at least 3 explanatory paragraphs."
    )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.8,
        max_tokens=800,
    )
    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable before running.")

    client = OpenAI(api_key=api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.seed(RANDOM_SEED)
    pairs: list[dict] = []

    topic_pool = (TOPICS * ((N_SAMPLES // len(TOPICS)) + 1))[:N_SAMPLES]
    random.shuffle(topic_pool)

    print(f"Generating {N_SAMPLES} instruction pairs in the domain: {DOMAIN}\n")
    for i, topic in enumerate(topic_pool, start=1):
        print(f"  [{i:02d}/{N_SAMPLES}] topic: {topic} ... ", end="", flush=True)
        try:
            pair = generate_pair(client, topic)
            pairs.append(pair)
            print("ok")
        except Exception as exc:
            print(f"ERROR: {exc}")

    if len(pairs) < 50:
        raise RuntimeError(f"Only {len(pairs)} pairs generated. Minimum required is 50.")

    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test.jsonl")

    for path, subset in [(train_path, train_pairs), (test_path, test_pairs)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"\nDataset saved to '{OUTPUT_DIR}/':\n"
        f"  train : {len(train_pairs)} pairs → {train_path}\n"
        f"  test  : {len(test_pairs)} pairs → {test_path}\n"
    )


if __name__ == "__main__":
    main()
