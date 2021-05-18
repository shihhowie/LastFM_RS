import os

from app import db
from sqlalchemy.dialects.postgresql import JSON


class Embedding(db.Model):
    __table_name__ = "user_embeddings"

    id = db.Column(db.Integer, primary_key=True)
    data_id = db.Column(db.Integer)
    user_embeddings = db.Column(JSON)

    def __init__(self, data_id, user_embeddings):
        self.data_id = data_id
        self.user_embeddings = user_embeddings

    def __repr__(self):
        return f"<id {self.id}>"
