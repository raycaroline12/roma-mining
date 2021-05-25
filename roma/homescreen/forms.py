from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(label='', required=True)
    file.widget.attrs.update({'class':'form-control form-control-lg'})