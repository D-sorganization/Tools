import urllib.request
try:
    print(urllib.request.urlopen("https://pypi.org/simple/kiwisolver/").info().get_content_type())
except Exception as e:
    print(e)
