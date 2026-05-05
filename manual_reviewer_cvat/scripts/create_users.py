"""Bootstrap CVAT users + project for the review pass.

Two responsibilities:
1. Create the 5 reviewer accounts (idempotent — skips existing usernames).
2. Optionally (`--create-project`) create the CVAT project with the label
   palette read from the dataset's classes.txt.

CVAT — unlike Label Studio Community — accepts a password on the user-create
REST call, so no Django-ORM workaround is needed. We still verify the user
can log in afterwards as a sanity check.

Usage:
    python -m manual_reviewer_cvat.scripts.create_users \\
        --cvat-url http://127.0.0.1:8081 \\
        --admin-user admin --admin-pass <pw> \\
        --reviewers pavan,sree,raj,sathish,deepak \\
        --default-password change-me-on-first-login

    python -m manual_reviewer_cvat.scripts.create_users --create-project \\
        --cvat-url http://127.0.0.1:8081 \\
        --admin-user admin --admin-pass <pw> \\
        --project-name "Datatang Diverse 1000" \\
        --classes-file output/dataset_selection/datatang_diverse_1000/yolo/classes.txt

NOT YET IMPLEMENTED — this is a stub. Body to be filled with cvat-sdk calls
once the user has approved the plan in README.md.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "Stub — see README.md step 2 & 3. Body will use cvat_sdk.api_client to:\n"
        "  - POST /api/auth/register for each reviewer (idempotent)\n"
        "  - GET /api/users/?search=<u> to verify creation\n"
        "  - POST /api/projects with labels=[{name,color} for each line in classes.txt]\n"
        "  - print PROJECT_ID=<n> on stdout for downstream scripts to capture"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--cvat-url", required=True)
    p.add_argument("--admin-user", required=True)
    p.add_argument("--admin-pass", required=True)
    p.add_argument("--reviewers", help="comma-separated usernames")
    p.add_argument("--default-password")
    p.add_argument("--create-project", action="store_true")
    p.add_argument("--project-name")
    p.add_argument("--classes-file")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
