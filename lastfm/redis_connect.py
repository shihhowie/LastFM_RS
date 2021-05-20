import os

import redis

r = redis.from_url(os.environ.get("REDIS_URL"))


def store_user_embedding_to_redis(embedding, model_name, data_id):
    r = redis.Redis()
    with r.pipeline() as pipe:
        for uid, u_embedding in embedding.items():
            key = f"{model_name}:window_{data_id}:uid_{uid}"
            if r.exists(key):
                r.delete(key)
            r.rpush(key, *u_embedding)
        pipe.execute()


def store_artist_embedding_to_redis():

    curr_path = os.path.dirname(os.path.abspath(__file__))

    with open(f"{curr_path}/embeddings/artist_embeddings.json") as f:
        aid_embedding = json.load(f)
    with r.pipeline() as pipe:
        for aid, a_embedding in aid_embedding.items():
            r.rpush(f"aid:{aid}", *a_embedding)
        pipe.execute()
