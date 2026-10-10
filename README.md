# NexoRobot

**Nexo** is a self-contained, browser-based 3D playground for a humanoid robot. The published site needs no build step, backend, or third-party runtime assets: the complete experience is in `index.html`.

## GitHub Pages

The repository includes `.github/workflows/deploy-pages.yml`, which deploys the static entry point from `main` with the official GitHub Pages Actions workflow.

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions** as the source.
3. Merge or push this change to `main`. The **Deploy GitHub Pages** workflow will publish it. You can also run it manually from the **Actions** tab.

Once deployed, the project site is available at:

- <https://ojperdomoc.github.io/NexoRobot/>

The old `Nexo.html` URL remains as a lightweight redirect so existing links continue to reach the site.

### Local preview

Serve the repository directory over HTTP rather than opening the file directly:

```bash
python3 -m http.server 8080
```

Then visit <http://localhost:8080/>. A secure HTTPS origin, such as GitHub Pages, is required for the Meta Quest/WebXR experience.

## Interaction

- Use the action buttons or keys **1–5** to interact with Nexo.
- Use **W A S D** or the arrow keys to move.
- Click the ground to choose a destination, and use the command box for instructions such as `sígueme`, `baila`, or `trae el núcleo`.

## Deployment notes

- `index.html` is the canonical Pages entry point.
- `.nojekyll` prevents accidental Jekyll processing if the Pages publishing mode is ever changed.
- The sharing QR/link is derived from the current HTTPS URL, so it works for the GitHub Pages project path and custom domains without a hard-coded tunnel URL.
