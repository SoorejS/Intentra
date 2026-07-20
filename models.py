from sqlalchemy import Column, Integer, String, Text, DateTime
from database import Base
import datetime
import json

class Generation(Base):
    __tablename__ = "generations"

    id = Column(Integer, primary_key=True, index=True)
    objective = Column(String, index=True)
    domain_hint = Column(String, nullable=True)
    schema_json = Column(Text)
    dataset_json = Column(Text)
    evaluation_json = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    @property
    def schema(self):
        return json.loads(self.schema_json)
        
    @property
    def dataset(self):
        return json.loads(self.dataset_json)
        
    @property
    def evaluation(self):
        return json.loads(self.evaluation_json)
