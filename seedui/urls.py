"""URL routes for the seed UI."""

from django.urls import path

from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.register_seed_view, name="register-seed"),
    path("identify/", views.identify_seed_view, name="identify-seed"),
    path("seed/<slug:seed_id>/", views.seed_detail_view, name="seed-detail"),
]
