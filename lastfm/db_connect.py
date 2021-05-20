import os

from app import db
from lastfm.embedding import UserEmbedding


def write_user_embedding(model_name="baseline"):
    from lastfm.tracker import load_user_embedding_json

    for i, embedding in load_user_embedding_json(model_name):
        new = UserEmbedding(i, model_name, embedding)
        db.session.add(new)
    db.session.commit()


def get_user_embedding(model_name="baseline"):
    embeddings = UserEmbedding.query.filter_by(model_name="baseline")
    for row in embeddings:
        # print(row.data_id)
        yield row.data_id, row.user_embeddings


if __name__ == "__main__":
    get_user_embedding()
