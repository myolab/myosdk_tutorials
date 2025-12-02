# MyoSDK C3D Retargeting Gradio App

This directory contains a Gradio web application that runs the MyoSDK C3D retargeting tutorial workflow from `tutorial_internal.ipynb`.

## What This App Does

The app provides a user-friendly interface to:
1. Upload C3D motion capture files
2. Upload markerset XML files (describing marker placement on the body)
3. Retarget the motion data onto a 3D character using MyoSDK
4. Download the resulting motion data as an NPY file

## Prerequisites

- **Python 3.8 or higher**
- **MyoSDK API key** from [MyoLab](https://dev.myolab.ai)
- **UV package manager** (recommended) or pip

## Installation with UV

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust.

### 1. Install UV

If you don't have UV installed, you can install it using one of these methods:

**Using pip:**
```bash
pip install uv
```

**Using Homebrew (macOS):**
```bash
brew install uv
```

**Using curl (Linux/macOS):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Using PowerShell (Windows):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies

Navigate to the `grApps` directory and install dependencies:

```bash
cd grApps
uv pip install gradio myosdk
```

Or if you prefer to use a virtual environment:

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install gradio myosdk
```

### 3. Alternative: Install with pip

If you prefer not to use UV:

```bash
pip install gradio myosdk
```

## Running the App
Simply run the app and enter your API key in the web interface:

```bash
python app_c3d_retarget.py
```

The app will be available at `http://localhost:7860` by default.

## Usage

1. **Start the app** using one of the methods above
2. **Open your browser** and navigate to `http://localhost:7860`
3. **Enter your API key** (or leave empty if set as environment variable)
4. **Upload your C3D file** - your motion capture data file
5. **Upload your markerset XML file** - describes where markers are placed on the body (e.g., `cmu_markerset.xml`)
6. **Click "Run Retargeting"** - the app will:
   - Upload both files to MyoSDK
   - Start a retargeting job
   - Wait for the job to complete (this may take several minutes)
   - Download the result
7. **Download the result** - the output NPY file will be available for download

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

- The retargeting process typically takes several minutes
- The output file is in NPY format (NumPy array) containing the retargeted motion data
- The app creates temporary files that are cleaned up automatically
- For production use, consider adding authentication and rate limiting

