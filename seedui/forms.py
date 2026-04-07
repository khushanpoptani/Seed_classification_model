"""Forms for registering and identifying seeds through the Django UI."""

from django import forms


class SeedRegistrationForm(forms.Form):
    name = forms.CharField(max_length=120, label="Seed Name")
    species = forms.CharField(max_length=120, required=False, label="Species")
    source = forms.CharField(max_length=160, required=False, label="Source")
    notes = forms.CharField(
        required=False,
        label="Notes",
        widget=forms.Textarea(attrs={"rows": 4, "placeholder": "Capture notes, sample batch, background, etc."}),
    )
    front = forms.ImageField(label="Front View")
    back = forms.ImageField(label="Back View")
    left = forms.ImageField(label="Left View")
    right = forms.ImageField(label="Right View")
    top = forms.ImageField(label="Top View")
    bottom = forms.ImageField(label="Bottom View")


class SeedIdentificationForm(forms.Form):
    image = forms.ImageField(label="Query Seed Image")
    top_k = forms.IntegerField(min_value=1, max_value=10, initial=5, label="Top Matches")
