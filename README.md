# haiweng05.github.io

Personal homepage for GitHub Pages.

## Local preview

Run a local static server from the repository root:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8000/
```

The `notes/` section uses Docsify, so preview it through the local server rather than opening the HTML file directly.

If port `8000` is already occupied, use another port, for example:

```powershell
python -m http.server 8001 --bind 127.0.0.1
```
