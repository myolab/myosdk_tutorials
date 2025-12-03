## MyoSapiens SDK (MyoSDK) Tutorials

This repository contains example scripts, apps, and data for learning how to use the  MyoSapiens SDK (MyoSDK) for motion capture retargeting and visualization.

### Contents
- **`python/`**: Jupyter tutorials and supporting Python assets.
- **`grApps/`**: Example graphical/CLI applications in [gradio](https://www.gradio.app/) for working with C3D data and retargeting.
- **`data/`**: Sample C3D motion capture files used by the tutorials and apps.
- **`markersets/`**: Marker set definition files (e.g., CMU marker configuration).

### Getting Started
#### Open the tutorials
Run`python/tutorial.ipynb` to step through example workflows.
#### Or, Explore the apps
Go in `grApps/` for end-to-end C3D retargeting examples via gradio UI.

Install Python dependencies** using
```bash
pip install -r grApps/requirements.txt
python grApps/app_c3d_retarget.py
```

