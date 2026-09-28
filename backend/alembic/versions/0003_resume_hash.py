"""Add exact uploaded-resume hash for duplicate detection.

Revision ID: 0003_resume_hash
Revises: 0002_v14_audit_evaluation
"""

from alembic import op
import sqlalchemy as sa


revision = "0003_resume_hash"
down_revision = "0002_v14_audit_evaluation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "candidates",
        sa.Column(
            "resume_hash",
            sa.String(length=64),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_candidates_resume_hash",
        "candidates",
        ["resume_hash"],
        unique=False,
    )

    op.create_unique_constraint(
        "uq_candidates_owner_resume_hash",
        "candidates",
        ["owner_id", "resume_hash"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_candidates_owner_resume_hash",
        "candidates",
        type_="unique",
    )

    op.drop_index(
        "ix_candidates_resume_hash",
        table_name="candidates",
    )

    op.drop_column(
        "candidates",
        "resume_hash",
    )
