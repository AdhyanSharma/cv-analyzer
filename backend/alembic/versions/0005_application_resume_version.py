"""Bind applications to the resume version used for screening.

Revision ID: 0005_application_resume_version
Revises: 0004_resume_versions
"""

from alembic import op
import sqlalchemy as sa


revision = "0005_application_resume_version"
down_revision = "0004_resume_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "applications",
        sa.Column(
            "resume_version_id",
            sa.String(length=36),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_applications_resume_version",
        "applications",
        ["resume_version_id"],
        unique=False,
    )

    op.create_foreign_key(
        "fk_applications_resume_version",
        "applications",
        "resume_versions",
        ["resume_version_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Existing applications should point to the candidate's
    # currently active resume version.
    op.execute(
        sa.text(
            """
            UPDATE applications AS a
            SET resume_version_id = rv.id
            FROM resume_versions AS rv
            WHERE rv.candidate_id = a.candidate_id
              AND rv.is_current = TRUE
            """
        )
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_applications_resume_version",
        "applications",
        type_="foreignkey",
    )

    op.drop_index(
        "ix_applications_resume_version",
        table_name="applications",
    )

    op.drop_column(
        "applications",
        "resume_version_id",
    )
