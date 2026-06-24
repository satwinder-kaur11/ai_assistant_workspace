"""
app/db/init_db.py

A startup script that connects to SQLite and creates all the tables defined in models.py.
"""

from app.db.session import engine, Base
# Import models to ensure they are registered with Base before creating tables
from app.db import models

def init_db():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

if __name__ == "__main__":
    init_db()
