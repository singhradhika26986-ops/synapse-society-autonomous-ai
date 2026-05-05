"""Deployment-friendly FastAPI entrypoint.

Platforms and examples often look for `app.py:app`. Re-export the main
production application here so the repository stays easy to deploy without
changing the canonical production implementation.
"""

from production_app import app
