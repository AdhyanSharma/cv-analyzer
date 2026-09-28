"""Add candidate resume version history.

Revision ID: 0004_resume_versions
Revises: 0003_resume_hash
"""

from alembic import op
import sqlalchemy as sa
import uuid


revision = "0004_resume_versions"
down_revision = "0003_resume_hash"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "resume_versions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "candidate_id",
            sa.String(length=36),
            sa.ForeignKey("candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "version_number",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "resume_filename",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "resume_text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "resume_hash",
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column(
            "is_current",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.String(length=40),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.String(length=40),
            nullable=False,
        ),
    )

    op.create_index(
        "ix_resume_versions_candidate",
        "resume_versions",
        ["candidate_id"],
    )

    op.create_index(
        "ix_resume_versions_hash",
        "resume_versions",
        ["resume_hash"],
    )

    op.create_unique_constraint(
        "uq_resume_versions_candidate_version",
        "resume_versions",
        ["candidate_id", "version_number"],
    )

    op.create_unique_constraint(
        "uq_resume_versions_candidate_hash",
        "resume_versions",
        ["candidate_id", "resume_hash"],
    )

    op.create_index(
        "uq_resume_versions_current_candidate",
        "resume_versions",
        ["candidate_id"],
        unique=True,
        postgresql_where=sa.text("is_current = true"),
    )

    # Backfill existing candidates as Resume v1.
    bind = op.get_bind()

    candidates = sa.table(
        "candidates",
        sa.column("id", sa.String()),
        sa.column("resume_filename", sa.String()),
        sa.column("resume_text", sa.Text()),
        sa.column("resume_hash", sa.String()),
        sa.column("created_at", sa.String()),
        sa.column("updated_at", sa.String()),
    )

    versions = sa.table(
        "resume_versions",
        sa.column("id", sa.String()),
        sa.column("candidate_id", sa.String()),
        sa.column("version_number", sa.Integer()),
        sa.column("resume_filename", sa.String()),
        sa.column("resume_text", sa.Text()),
        sa.column("resume_hash", sa.String()),
        sa.column("is_current", sa.Boolean()),
        sa.column("created_at", sa.String()),
        sa.column("updated_at", sa.String()),
    )

    rows = bind.execute(
        sa.select(
            candidates.c.id,
            candidates.c.resume_filename,
            candidates.c.resume_text,
            candidates.c.resume_hash,
            candidates.c.created_at,
            candidates.c.updated_at,
        )
    ).fetchall()

    for row in rows:
        bind.execute(
            versions.insert().values(
                id=str(uuid.uuid4()),
                candidate_id=row.id,
                version_number=1,
                resume_filename=row.resume_filename,
                resume_text=row.resume_text,
                resume_hash=row.resume_hash,
                is_current=True,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        )


def downgrade() -> None:
    op.drop_index(
        "uq_resume_versions_current_candidate",
        table_name="resume_versions",
    )

    op.drop_constraint(
        "uq_resume_versions_candidate_hash",
        "resume_versions",
        type_="unique",
    )

    op.drop_constraint(
        "uq_resume_versions_candidate_version",
        "resume_versions",
        type_="unique",
    )

    op.drop_index(
        "ix_resume_versions_hash",
        table_name="resume_versions",
    )

    op.drop_index(
        "ix_resume_versions_candidate",
        table_name="resume_versions",
    )

    op.drop_table("resume_versions")
