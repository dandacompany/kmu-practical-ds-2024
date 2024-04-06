from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from sqlalchemy import exc
import pandas as pd

class DB:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DB, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance
    
    def __del__(self):
        self.session.close()

    def _initialize(self):
        self.engine = create_engine('postgresql+psycopg2://postgres:postgres@postgresql:5432/public')
        self.session = sessionmaker(bind=self.engine)()

    def execute(self, query):
        try:
            result = self.session.execute(text(query))
        except exc.SQLAlchemyError:
            self.session = sessionmaker(bind=self.engine)()
            result = self.session.execute(text(query))
        return pd.DataFrame(result.mappings().all())
    
    def load(self, table_name: str):
        try:
            result = self.session.execute(f"SELECT * FROM {table_name}")
        except exc.SQLAlchemyError:
            self.session = sessionmaker(bind=self.engine)()
            result = self.session.execute(f"SELECT * FROM {table_name}")
        return pd.DataFrame(result.mappings().all())
    
    def insert(self, df: pd.DataFrame, table_name: str):
        try:
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
        except exc.SQLAlchemyError:
            self.session = sessionmaker(bind=self.engine)()
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            
    def delete(self, table_name: str):
        try:
            self.session.execute(f"DROP TABLE IF EXISTS {table_name}")
        except exc.SQLAlchemyError:
            self.session = sessionmaker(bind=self.engine)()
            self.session.execute(f"DROP TABLE IF EXISTS {table_name}")


if __name__ == "__main__":
    db = DB()
    print(db.execute("SELECT 1"))


