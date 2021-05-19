import os

from sqlalchemy.dialects.postgresql import JSON
from flask_sqlalchemy import SQLAlchemy
from app import db


class UserEmbedding(db.Model):
    __table_name__ = "user_embeddings"

    id = db.Column(db.Integer, primary_key=True)
    data_id = db.Column(db.Integer)
    model_name = db.Column(db.String())
    user_embeddings = db.Column(JSON)

    def __init__(self, data_id, model_name, user_embeddings):
        self.data_id = data_id
        self.model_name = model_name
        self.user_embeddings = user_embeddings

    def __repr__(self):
        return f"<id {self.id}>"
