from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from database import Base
import datetime
import json
import uuid

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
    dataset_versions = relationship("DatasetVersion", back_populates="project")
    training_runs = relationship("TrainingRun", back_populates="project")
    optimization_cycles = relationship("OptimizationCycle", back_populates="project")


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
        return json.loads(self.schema_json) if self.schema_json else {}
        
    @property
    def dataset(self):
        return json.loads(self.dataset_json) if self.dataset_json else []
        
    @property
    def evaluation(self):
        return json.loads(self.evaluation_json) if self.evaluation_json else {}


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String) # generate, refine, train, optimize
    examples_count = Column(Integer, default=0)
    tokens_estimated = Column(Integer, default=0)
    cost_estimated_usd = Column(Text, default="0.00")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="usage_logs")


# ─── V2 Closed-Loop Models ───────────────────────────────────────────────────

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    version_number = Column(Integer, default=1)
    parent_version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=True)
    total_examples = Column(Integer, default=0)
    canonical_count = Column(Integer, default=0)
    boundary_count = Column(Integer, default=0)
    adversarial_count = Column(Integer, default=0)
    generated_by = Column(String, default="intentra_v2") # initial, targeted_boundary, hard_negative, user_edit
    generation_reason = Column(Text, nullable=True)
    status = Column(String, default="draft") # draft, training, evaluated, promoted, rejected
    dataset_json = Column(Text) # JSON array of {text, label, type, ...}
    schema_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    promoted_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="dataset_versions")
    parent_version = relationship("DatasetVersion", remote_side=[id])
    training_runs = relationship("TrainingRun", back_populates="dataset_version")

    @property
    def dataset(self):
        return json.loads(self.dataset_json) if self.dataset_json else []

    @property
    def schema(self):
        return json.loads(self.schema_json) if self.schema_json else {}


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    dataset_version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=False)
    model_name = Column(String, default="distilbert-base-uncased")
    model_type = Column(String, default="sequence_classification")
    framework = Column(String, default="transformers") # transformers, sklearn_fast
    seed = Column(Integer, default=42)
    epochs = Column(Integer, default=3)
    training_time_seconds = Column(Float, default=0.0)
    training_status = Column(String, default="pending") # pending, training, completed, failed
    artifact_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="training_runs")
    dataset_version = relationship("DatasetVersion", back_populates="training_runs")
    evaluation_runs = relationship("EvaluationRun", back_populates="training_run")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=False)
    test_set_version = Column(String, default="holdout_v1")
    split_type = Column(String, default="val") # val, test
    accuracy = Column(Float, default=0.0)
    macro_f1 = Column(Float, default=0.0)
    weighted_f1 = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    hard_negative_accuracy = Column(Float, default=0.0)
    boundary_accuracy = Column(Float, default=0.0)
    adversarial_accuracy = Column(Float, default=0.0)
    per_class_metrics_json = Column(Text, default="{}") # {class: {f1, precision, recall, support}}
    confusion_matrix_json = Column(Text, default="{}") # {classes: [...], matrix: [[...]]}
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    training_run = relationship("TrainingRun", back_populates="evaluation_runs")
    classification_errors = relationship("ClassificationError", back_populates="evaluation_run")

    @property
    def per_class_metrics(self):
        return json.loads(self.per_class_metrics_json) if self.per_class_metrics_json else {}

    @property
    def confusion_matrix(self):
        return json.loads(self.confusion_matrix_json) if self.confusion_matrix_json else {}


class ClassificationError(Base):
    __tablename__ = "classification_errors"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    evaluation_run_id = Column(String, ForeignKey("evaluation_runs.id"), nullable=False)
    input_text = Column(Text, nullable=False)
    expected_label = Column(String, nullable=False)
    predicted_label = Column(String, nullable=False)
    confidence = Column(Float, default=0.0)
    error_type = Column(String, default="class_confusion") # class_confusion, boundary_failure, hard_negative_failure, low_confidence, adversarial_failure
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    evaluation_run = relationship("EvaluationRun", back_populates="classification_errors")


class OptimizationCycle(Base):
    __tablename__ = "optimization_cycles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    base_dataset_version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=True)
    resulting_dataset_version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=True)
    evaluation_run_id = Column(String, ForeignKey("evaluation_runs.id"), nullable=True)
    target_problem = Column(Text, nullable=True) # e.g. "Confusion between refund_request and cancel_order (18 errors)"
    examples_generated = Column(Integer, default=0)
    examples_accepted = Column(Integer, default=0)
    examples_rejected = Column(Integer, default=0)
    baseline_f1 = Column(Float, default=0.0)
    resulting_f1 = Column(Float, default=0.0)
    improvement_delta = Column(Float, default=0.0)
    status = Column(String, default="running") # running, promoted, rejected, failed
    rejection_reason = Column(Text, nullable=True)
    telemetry_json = Column(Text, default="{}") # {tokens, cost_usd, training_time_s, eval_time_s}
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="optimization_cycles")

    @property
    def telemetry(self):
        return json.loads(self.telemetry_json) if self.telemetry_json else {}
