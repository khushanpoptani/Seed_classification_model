"""WSGI config for the seed web project."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "seedweb.settings")

application = get_wsgi_application()
