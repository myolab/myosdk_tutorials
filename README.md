# MyoSapiens SDK (MyoSDK) Tutorials

This repository contains example scripts, apps, and data for learning how to use the  MyoSapiens SDK (MyoSDK) for motion capture retargeting and visualization.

For details on the SDK, please visit [docs.myolab.ai](https://docs.myolab.ai)

## Contents
- **`python/`**: Jupyter tutorials and supporting Python assets.
- **`grApps/`**: Example graphical/CLI applications in [gradio](https://www.gradio.app/) for working with C3D data and retargeting. See [Readme](./grApps/README.md).
- **`data/`**: Sample C3D motion capture files (e.g., CMU) used by the tutorials and apps.
- **`markersets/`**: Marker set definition files (e.g., CMU marker configuration). See [Readme](./markersets/Readme.md).

## Getting Started
### Open the tutorials
Install Python dependencies using
```bash
pip install myosdk jupyter ipykernel
```
Run`python/tutorial.ipynb` to step through example workflows.
### Or, Explore the apps
Go in `grApps/` for end-to-end C3D retargeting examples via gradio UI.

Install Python dependencies using
```bash
pip install -r grApps/requirements.txt
python grApps/app_c3d_retarget.py
```
## Example Files

Example files are available in the repository:
- **C3D file**: `../data/35_30.c3d`
- **Markerset**: `../markersets/cmu_markerset.xml`

## Troubleshooting

### API Key Issues
- Make sure your API key is correct and active
- Check that you have an active MyoLab account
- Verify the API key is set correctly (environment variable or entered in the app)

### File Upload Issues
- Ensure your C3D file is a valid motion capture file
- Verify your markerset XML file is properly formatted
- Check that both files are compatible (markerset matches the markers in your C3D file)

### Job Failures
- The retargeting process may fail if the C3D and markerset files are incompatible
- Check the status message in the app for specific error details
- Ensure your files are not corrupted

### Network Issues
- Make sure you have a stable internet connection
- The app needs to communicate with MyoSDK servers
- Large files may take time to upload

## Development

To modify the app, edit `app_c3d_retarget.py`. The app uses:
- **Gradio** for the web interface
- **MyoSDK** for the retargeting API

## Additional Resources

- [MyoSDK Documentation](https://docs.myolab.ai/docs/myosdk/getting-started/api-key)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [UV Documentation](https://github.com/astral-sh/uv)

## Notes

- The retargeting process typically might several seconds depending on the lenght of the motion file
- The output file is in NPZ format (NumPy binary) containing the retargeted motion data and joint information
- The app creates temporary files that are cleaned up automatically
- For production use, consider adding authentication and rate limiting

