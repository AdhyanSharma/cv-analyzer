"""Add exact resume-file hash for duplicate detection.

Revision ID: 0002_resume_hash
Revises: 0001_initial_platform
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_resume_hash"
down_revision = "0001_initial_platform"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "candidates",
        sa.Column("resume_hash", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "idx_candidates_resume_hash",
        "candidates",
        ["resume_hash"],
        unique=False,
    )
    op.create_index(
        "uq_candidates_owner_resume_hash",
        "candidates",
        ["owner_id", "resume_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_candidates_owner_resume_hash", table_name="candidates")
    op.drop_index("idx_candidates_resume_hash", table_name="candidates")
    op.drop_column("candidates", "resume_hash")
