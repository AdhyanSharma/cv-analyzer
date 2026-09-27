"""Add V14 audit logs and evaluation runs."""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0002_v14_audit_evaluation"
down_revision: Union[str, None] = "0001_initial_platform"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("entity_id", sa.String(length=36), server_default="", nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_owner_id", "audit_logs", ["owner_id"], unique=False)
    op.create_index("ix_audit_logs_owner_created", "audit_logs", ["owner_id", "created_at"], unique=False)
    op.create_index("ix_audit_logs_entity", "audit_logs", ["entity_type", "entity_id"], unique=False)

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evaluation_runs_owner_id", "evaluation_runs", ["owner_id"], unique=False)
    op.create_index("ix_evaluation_runs_job_id", "evaluation_runs", ["job_id"], unique=False)
    op.create_index("ix_evaluation_runs_owner_job", "evaluation_runs", ["owner_id", "job_id"], unique=False)
    op.create_index("ix_evaluation_runs_created", "evaluation_runs", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_evaluation_runs_created", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_owner_job", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_job_id", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_owner_id", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
    op.drop_index("ix_audit_logs_entity", table_name="audit_logs")
    op.drop_index("ix_audit_logs_owner_created", table_name="audit_logs")
    op.drop_index("ix_audit_logs_owner_id", table_name="audit_logs")
    op.drop_table("audit_logs")
