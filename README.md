[![PyPI](https://img.shields.io/pypi/v/myosdk)](https://pypi.org/project/myosdk)
[![Docs](https://img.shields.io/badge/Docs-Online-blue)](https://docs.myolab.ai)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IHmXo8NdSz0Jrkt8mfI9caIn8ajc-h5x)


# MyoSapiens SDK (MyoSDK) Tutorials

This repository contains example scripts, apps, and data for learning how to use the MyoSapiens SDK (MyoSDK) for motion capture retargeting (converting motion data to work with different character models) and visualization.

For details on the SDK, please visit [docs.myolab.ai](https://docs.myolab.ai)

## Contents
- **`python/`**: Jupyter notebook tutorials (interactive Python notebooks) and supporting Python assets.
- **`grApps/`**: Example web applications built with [Gradio](https://www.gradio.app/) (a Python library for creating user interfaces) for working with C3D motion capture files and retargeting. See [Readme](./grApps/README.md).
- **`data/`**: Sample C3D motion capture files (C3D is a standard format for storing motion capture data, e.g., CMU dataset) used by the tutorials and apps.
- **`markersets/`**: Marker set definition files (XML files that describe which markers are used in your motion capture data, e.g., CMU marker configuration). See [Readme](./markersets/Readme.md).

## Getting Started

You can either follow the interactive tutorials or use the web-based application. Choose the option that works best for you.

### Option 1: Interactive Tutorials (Recommended for learning)

1. Install the required Python packages:
```bash
pip install myosdk jupyter ipykernel git+https://github.com/Vittorio-Caggiano/metrabs.git
```

2. Open the tutorial notebook:
   - If you have Jupyter installed, you can run: `jupyter notebook python/tutorial.ipynb`
   - Or open `python/tutorial.ipynb` in your preferred notebook environment (Jupyter Lab, VS Code, etc.)
   - The notebook will guide you through example workflows step by step

### Option 2: Web Application (Quick start)

1. Navigate to the `grApps/` directory and install dependencies:
```bash
pip install -r grApps/requirements.txt
```

2. Run the application:
```bash
python grApps/app_retarget.py
```

3. The app will start a local web server. Open the URL shown in your terminal (usually `http://127.0.0.1:7860`) in your web browser to use the graphical interface.
## Example Files

Example files are included in this repository to help you get started:
- **C3D file**: `data/35_30.c3d` - A sample motion capture file
- **Markerset**: `markersets/cmu_markerset.xml` - The corresponding marker configuration file that describes the markers used in the C3D file

## Troubleshooting

### API Key Issues
- Make sure your API key is correct and active
- Check that you have an active MyoLab account (sign up at [docs.myolab.ai](https://docs.myolab.ai) if needed)
- Verify the API key is set correctly:
  - In the web app: Enter it in the API key field
  - In Python scripts: Set it as an environment variable or pass it directly in your code

### File Upload Issues
- Ensure your C3D file is a valid motion capture file (not corrupted)
- Verify your markerset XML file is properly formatted (follow the examples in `markersets/`)
- Check that both files are compatible: the markerset must describe the same markers that are present in your C3D file

### Job Failures
- The retargeting process may fail if the C3D and markerset files are incompatible
- Check the status message in the app for specific error details
- Ensure your files are not corrupted

### Network Issues
- Make sure you have a stable internet connection
- The app needs to communicate with MyoSDK servers
- Large files may take time to upload

## Development

To modify the app, edit `app_retarget.py`. The app uses:
- **Gradio** for the web interface
- **MyoSDK** for the retargeting API

## Additional Resources

- [MyoSDK Documentation](https://docs.myolab.ai/docs/myosdk/getting-started/api-key)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [UV Documentation](https://github.com/astral-sh/uv)

## Notes

- The retargeting process typically takes several seconds depending on the length of the motion file
- The output file is in NPZ format (NumPy binary format, which you can load in Python using `numpy.load()`) containing the retargeted motion data and joint information
- The app creates temporary files that are cleaned up automatically
- For production use, consider adding authentication and rate limiting

