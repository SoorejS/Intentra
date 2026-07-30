from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from database import Base
import datetime
import json

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_pro = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    projects = relationship("Project", back_populates="owner")
    generations = relationship("Generation", back_populates="user")
    usage_logs = relationship("UsageLog", back_populates="user")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="projects")
    generations = relationship("Generation", back_populates="project")


class Generation(Base):
    __tablename__ = "generations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    objective = Column(String, index=True)
    domain_hint = Column(String, nullable=True)
    schema_json = Column(Text)
    dataset_json = Column(Text)
    evaluation_json = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="generations")
    project = relationship("Project", back_populates="generations")

    @property
    def schema(self):
        return json.loads(self.schema_json)
        
    @property
    def dataset(self):
        return json.loads(self.dataset_json)
        
    @property
    def evaluation(self):
        return json.loads(self.evaluation_json)


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String) # generate, refine
    examples_count = Column(Integer, default=0)
    tokens_estimated = Column(Integer, default=0)
    cost_estimated_usd = Column(Text, default="0.00")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="usage_logs")
