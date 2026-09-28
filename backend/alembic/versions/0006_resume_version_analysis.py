"""Store job-specific analysis snapshots for resume versions.

Revision ID: 0006_resume_version_analysis
Revises: 0005_application_resume_version
"""

from alembic import op
import sqlalchemy as sa


revision = "0006_resume_version_analysis"
down_revision = "0005_application_resume_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "resume_version_analyses",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "resume_version_id",
            sa.String(length=36),
            sa.ForeignKey("resume_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.String(length=36),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "analysis_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
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
        "ix_resume_version_analyses_version",
        "resume_version_analyses",
        ["resume_version_id"],
    )

    op.create_index(
        "ix_resume_version_analyses_job",
        "resume_version_analyses",
        ["job_id"],
    )

    op.create_unique_constraint(
        "uq_resume_version_analysis_version_job",
        "resume_version_analyses",
        ["resume_version_id", "job_id"],
    )

    # Backfill existing application analyses.
    #
    # Every application that already has a resume_version_id gets
    # its existing screening analysis preserved as a version/job
    # analysis snapshot.
    op.execute(
        sa.text(
            """
            INSERT INTO resume_version_analyses (
                id,
                resume_version_id,
                job_id,
                analysis_json,
                created_at,
                updated_at
            )
            SELECT
                gen_random_uuid()::text,
                a.resume_version_id,
                a.job_id,
                a.analysis_json,
                a.created_at,
                a.updated_at
            FROM applications AS a
            WHERE a.resume_version_id IS NOT NULL
            """
        )
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_resume_version_analysis_version_job",
        "resume_version_analyses",
        type_="unique",
    )

    op.drop_index(
        "ix_resume_version_analyses_job",
        table_name="resume_version_analyses",
    )

    op.drop_index(
        "ix_resume_version_analyses_version",
        table_name="resume_version_analyses",
    )

    op.drop_table("resume_version_analyses")
