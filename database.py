from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./intentra.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def auto_migrate_sqlite():
    """Ensure newly added columns exist in SQLite database tables."""
    try:
        with engine.connect() as conn:
            # Check columns in 'generations' table
            result = conn.execute(text("PRAGMA table_info(generations)"))
            existing_cols = {row[1] for row in result.fetchall()}
            
            if existing_cols: # Table exists
                if "user_id" not in existing_cols:
                    conn.execute(text("ALTER TABLE generations ADD COLUMN user_id INTEGER REFERENCES users(id)"))
                    print("[Database Migration] Added missing column 'user_id' to generations table.")
                if "project_id" not in existing_cols:
                    conn.execute(text("ALTER TABLE generations ADD COLUMN project_id INTEGER REFERENCES projects(id)"))
                    print("[Database Migration] Added missing column 'project_id' to generations table.")
                conn.commit()
    except Exception as e:
        print(f"[Database Migration] Warning: {e}")
