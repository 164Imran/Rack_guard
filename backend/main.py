"""Stable entrypoint that composes independent Rack Guardian modules."""

from .api import create_app

app = create_app()
