"""
Copyright (c) 2025 MyoLab, Inc. All Rights Reserved.

This software and associated documentation files (the "Software") are the intellectual property of MyoLab, Inc. Unauthorized copying, modification, distribution, or use of this code, in whole or in part, without express written permission from the copyright owner is strictly prohibited.


MyoSDK Retargeting App
"""

import os
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from metrabs_pytorch.scripts.run_video import run_metrabs_video
from myo_tools.mjs.marker.marker_api import get_marker_names
from myo_tools.utils.file_ops.dataframe_utils import from_array_to_dataframe
from myosdk import Client

PLOT_CONFIG = {
    "plot_bgcolor": "#0f172a",
    "paper_bgcolor": "#0f172a",
    "font": {"color": "#e2e8f0", "family": "Inter, system-ui, sans-serif"},
    "xaxis": {"gridcolor": "#1e293b", "linecolor": "#334155"},
    "yaxis": {"gridcolor": "#1e293b", "linecolor": "#334155"},
}

custom_css = """
.upload-box {
    border: 2px dashed #ccc;
    border-radius: 8px;
    padding: 30px;
    text-align: center;
    cursor: pointer;
}
.upload-box:hover {
    border-color: #666;
}
#file-upload {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0 !important;
}
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def draw_keypoints(frame, poses2d, radius=10):
    """
    frame: HxWx3 uint8
    poses2d: NxJx2 (N people, J joints)
    """
    for person in poses2d:
        for x, y in person:
            cv2.circle(frame, (int(x), int(y)), radius, (0, 255, 0), -1)
    return frame


def save_video_with_keypoints(results, output_video):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        output_video,
        fourcc,
        results[0]["fps"],
        (results[0]["frame_bgr"].shape[1], results[0]["frame_bgr"].shape[0]),
    )
    for res in results:
        frame = res["frame_bgr"]
        fps = res["fps"]
        poses2d = res["poses2d"]  # NxJx2
        frame = draw_keypoints(frame, poses2d)
        out.write(frame)
    out.release()
    return output_video


def update_display(files, str_msg="📁 Click to select XML files"):
    if files is None:
        return f'<div class="upload-box" onclick="document.querySelector(\'input[type=file]\').click()">{str_msg}</div>'

    # Handle single file (not a list) or empty list
    if not isinstance(files, list):
        files = [files]

    if len(files) == 0:
        return f'<div class="upload-box" onclick="document.querySelector(\'input[type=file]\').click()">{str_msg}</div>'

    # Handle both file objects (with .name attribute) and strings (file paths)
    filenames = []
    for f in files:
        if isinstance(f, str):
            # If it's a string, extract filename from path
            filenames.append(f.split("/")[-1])
        elif hasattr(f, "name"):
            # If it's a file object with .name attribute
            filenames.append(f.name.split("/")[-1])
        else:
            # Fallback: convert to string and extract filename
            filenames.append(str(f).split("/")[-1])

    file_list = "<br>".join([f"✓ {name}" for name in filenames])
    return f'<div class="upload-box" onclick="document.querySelector(\'input[type=file]\').click()">{file_list}<br><br>Click to reselect</div>'


def load_all_videos():
    video_dir = os.path.join(os.path.dirname(__file__), "../data")
    return [
        os.path.abspath(os.path.join(video_dir, f))
        for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]


# ------------------------------------------------------------
# Retargeting
# ------------------------------------------------------------
def run_retargeting_c3d(api_key, c3d_files, markerset_file):
    status = []
    output_files = []

    # Initial validation
    if not api_key:
        api_key = os.getenv("MYOSDK_API_KEY")
        if not api_key:
            yield (
                "❌ Error: API key missing",
                None,
                None,
                gr.update(value=[], visible=True),
                gr.update(visible=False),
            )
            return

    if markerset_file is None:
        yield (
            "❌ Error: Markerset XML file is required",
            None,
            None,
            gr.update(value=[], visible=True),
            gr.update(visible=False),
        )
        return

    try:
        # Initialize client
        status.append("🔹 Initializing MyoSDK client...")
        init_time = time.time()
        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=False
        )
        client = Client(api_key=api_key)

        status.append(
            f"🔹 MyoSDK client initialized in { time.time() - init_time:.2f} seconds"
        )
        init_time = time.time()
        # Upload markerset
        status.append("🔹 Uploading markerset file...")
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        )
        mk_asset = client.assets.upload_file(markerset_file.name)

        status.append(
            f"🔹 Markerset file uploaded in {time.time() - init_time:.2f} seconds"
        )
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        )

        mk_id = mk_asset["asset_id"]

        # Process each C3D file
        total_files = len(c3d_files)
        for idx, f in enumerate(c3d_files):
            status.append(
                f"➡ Processing file {idx + 1}/{total_files}: {os.path.basename(f)}"
            )

            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            )
            init_time = time.time()
            c3d_asset = client.assets.upload_file(f)
            status.append(
                f"\t🔹 C3D file uploaded in {time.time() - init_time:.2f} seconds"
            )
            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            )
            init_time = time.time()
            job = client.jobs.start_retarget(
                c3d_asset_id=c3d_asset["asset_id"],
                markerset_asset_id=mk_id,
            )

            status.append(
                f"\t🔹 Retargeting job started in {time.time() - init_time:.2f} seconds"
            )
            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            )
            init_time = time.time()
            result = client.jobs.wait(job["job_id"])

            status.append(
                f"\t🔹 Retargeting job completed in {time.time() - init_time:.2f} seconds"
            )
            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            )
            if result["status"] != "SUCCEEDED":
                status.append(f"\t❌ Failed retarget for {os.path.basename(f)}")
                continue

            status.append(f"\t✅ Retargeting completed for {os.path.basename(f)}")
            base = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(tempfile.gettempdir(), base + ".npy")
            client.assets.download(
                result["output"]["retarget_output_asset_id"], out_path
            )
            output_files.append(out_path)

        if not output_files:
            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            )
            return

        # Load angles from first output file
        status.append("🔹 Loading angle data...")
        yield "\n".join(status), None, None, gr.update(
            interactive=True, visible=True
        ), gr.update(visible=True)

        data = np.load(output_files[0])
        joint_angles = data["joint_angles_degrees"].squeeze()
        joint_names = data["joint_names"]

        df = pd.DataFrame(joint_angles, columns=[jn for jn in joint_names])
        df.insert(0, "frame", df.index)

        angle_list = list(df.columns[1:])
        initial_value = [angle_list[0]] if angle_list else []

        status.append("✅ Complete!")
        yield (
            "\n".join(status),
            gr.update(value=output_files, visible=True),
            df,
            gr.update(choices=angle_list, value=initial_value, visible=True),
            gr.update(visible=True),
        )

    except Exception as e:
        yield (
            f"❌ {e}",
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )


def run_retargeting_video(
    api_key,
    video_file="",
    model="metrabs",
):
    status = []

    # Initial validation
    if not api_key:
        api_key = os.getenv("MYOSDK_API_KEY")
        if not api_key:  # covers None, "", or other falsy values
            yield (
                "❌ Error: API key is missing or invalid",
                None,
                None,
                gr.update(visible=False),
                gr.update(visible=False),
                None,
            )
            raise ValueError("❌ Error: API key is missing or invalid")

    # Extract path from list if it's a list, otherwise use directly
    if isinstance(video_file, list):
        video_path = video_file[0] if len(video_file) > 0 else None
    else:
        video_path = video_file

    if (
        video_file is None
        or (isinstance(video_file, list) and len(video_file) == 0)
        or video_path is None
    ):
        yield (
            "❌ Error: No video file selected",
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            video_file,
        )
        return

    try:

        print("🔹 Pose Extraction from Video Started")
        status.append(
            "🔹 Pose Extraction from Video Started ... this may take a while depending on the video length."
        )
        init_time = time.time()
        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=False
        ), video_path

        results = list(
            run_metrabs_video(video_path=video_path, device=DEVICE, visualize=False)
        )

        markers = np.array([res["poses3d"] for res in results]).squeeze()
        fps = (
            results[0]["fps"] if results else 25.0
        )  # Default to 25 fps if not available

        video_with_keypoints = os.path.join(
            tempfile.gettempdir(), "video_with_keypoints.mp4"
        )
        save_video_with_keypoints(results, video_with_keypoints)

        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=True
        ), video_with_keypoints,
        status.append(
            f"🔹 Pose Extraction from Video Completed in {time.time() - init_time:.2f} seconds with {len(markers)} frames extracted ({((time.time() - init_time)/len(markers)):.2f} seconds per frame)"
        )
        print("🔹 Pose Extraction from Video Completed")
        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=False
        ), video_with_keypoints
        # Initialize client
        status.append("🔹 Initializing MyoSDK client...")
        init_time = time.time()
        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=False,
        ), video_with_keypoints
        client = Client(api_key=api_key)

        status.append(
            f"🔹 MyoSDK client initialized in { time.time() - init_time:.2f} seconds"
        )
        init_time = time.time()
        # Upload markerset
        status.append("🔹 Uploading markerset file...")
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False,
        ), video_with_keypoints

        markerset_file_name = "markersets/movi_metrabs_markerset.xml"

        mk_asset = client.assets.upload_file(markerset_file_name)

        status.append(
            f"🔹 Markerset file uploaded in {time.time() - init_time:.2f} seconds"
        )

        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        ), video_with_keypoints
        init_time = time.time()

        marker_names = get_marker_names(markerset_file_name)
        fn_parquet = os.path.join(tempfile.gettempdir(), "video_trackers.parquet")
        from_array_to_dataframe(markers, marker_names, fps, fn_parquet)
        markers_asset = client.assets.upload_file(fn_parquet, purpose="retarget")

        print("fn_parquet: ", fn_parquet)

        init_time = time.time()
        job = client.jobs.start_retarget(
            c3d_asset_id=markers_asset["asset_id"],
            markerset_asset_id=mk_asset["asset_id"],
        )

        status.append(
            f"\t🔹 Retargeting job started in {time.time() - init_time:.2f} seconds"
        )
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        ), video_with_keypoints
        init_time = time.time()
        result = client.jobs.wait(job["job_id"])

        status.append(
            f"\t🔹 Retargeting job completed in {time.time() - init_time:.2f} seconds"
        )
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        ), video_with_keypoints
        print("STATUS: ", result["status"])
        assert (
            result["status"] == "SUCCEEDED"
        ), f"Failed retarget for {os.path.basename(video_path)}"

        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(tempfile.gettempdir(), base + ".npy")
        client.assets.download(result["output"]["retarget_output_asset_id"], out_path)

        assert os.path.exists(
            out_path
        ), f"Failed to download retargeted data for {os.path.basename(video_path)}"

        # Load angles from first output file
        status.append("🔹 Loading angle data...")
        yield "\n".join(status), None, None, gr.update(
            interactive=True, visible=True
        ), gr.update(visible=True), video_with_keypoints

        data = np.load(out_path)
        joint_angles = data["joint_angles_degrees"].squeeze()
        joint_names = data["joint_names"]

        df = pd.DataFrame(joint_angles, columns=[jn for jn in joint_names])
        df.insert(0, "frame", df.index)

        angle_list = list(df.columns[1:])
        initial_value = [angle_list[0]] if angle_list else []

        status.append("✅ Complete!")
        yield (
            "\n".join(status),
            gr.update(value=out_path, visible=True),
            df,
            gr.update(choices=angle_list, value=initial_value, visible=True),
            gr.update(visible=True),
            video_with_keypoints,
        )

    except Exception as e:
        # Use video_path if defined, otherwise use video_file or None
        error_video = (
            video_path
            if "video_path" in locals()
            else (video_file if video_file else None)
        )
        yield (
            "\n".join(status + ["\n❌ Error: " + str(e)]),
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            error_video,
        )


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def update_plot(df, joints):
    if df is None or df.empty:
        return go.Figure()
    if not joints:
        return go.Figure()
    if not isinstance(joints, list):
        joints = [joints]

    fig = go.Figure()
    for j in joints:
        if j in df.columns:
            fig.add_trace(go.Scatter(x=df["frame"], y=df[j], mode="lines", name=j))

    fig.update_layout(
        title="Joint Angles",
        xaxis_title="Frame",
        yaxis_title="Angle Value",
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(color="#F0F0F0", family="Arial"),
        xaxis=dict(gridcolor="#444444", linecolor="#F0F0F0", tickcolor="#F0F0F0"),
        yaxis=dict(gridcolor="#444444", linecolor="#F0F0F0", tickcolor="#F0F0F0"),
        legend=dict(font=dict(color="#F0F0F0")),
    )
    return fig


with gr.Blocks(css=custom_css) as app:

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                """
                ## MyoSDK Retargeting
                <span style="color:#6b7280">Joint visualization & motion retargeting pipelines</span>

                This application allows you to retarget motion capture data to biomechanical models using MyoSDK's Kinesis engine.
                Upload C3D files or videos to extract joint angles using [Kinesis](https://myolab.ai/blog/myokinesis) and visualize motion data.
                """
            )
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="🔑 API Key",
                placeholder="Enter your MyoLab API key",
                type="password",
                info="Get your API key from https://dev.myolab.ai",
            )
    with gr.Tab("📊 Motion Capture Retargeting"):

        gr.Markdown(
            """
            Upload motion capture data in C3D format along with a markerset XML file to retarget the motion to a biomechanical model.
            The process will extract joint angles and generate visualizations of the motion data.
            """
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown(
                    """
                    **1. Upload a Markerset File**
                    <br>
                    <span style="color:#6b7280; font-size: 0.9em">
                    Upload an XML file that defines the marker set configuration.
                    This file specifies which markers are used and their anatomical locations.
                    See [Markerset Editor](https://markerset-editor.myolab.ai) for more details.
                    </span>
                    """
                )
                upload_markerset_display = gr.HTML(
                    '<div class="upload-box" onclick="document.querySelector(\'input[type=file]\').click()">📁 Click to select XML file</div>'
                )

                markerset = gr.File(
                    label=None,
                    file_types=[".xml"],
                    elem_id="file-upload",
                )

                markerset.change(
                    lambda files: update_display(files, "📁 Click to select XML file"),
                    [markerset],
                    upload_markerset_display,
                )

            with gr.Column(scale=2):
                gr.Markdown(
                    """
                    **2. Upload C3D Motion Capture File(s)**
                    <br>
                    <span style="color:#6b7280; font-size: 0.9em">
                    Upload one or more C3D files containing 3D marker trajectories from motion capture systems.
                    Multiple files can be processed in batch. Each file will be retargeted using the same markerset.
                    </span>
                    """
                )
                upload_c3d_display = gr.HTML(
                    '<div class="upload-box" onclick="document.querySelector(\'input[type=file]\').click()">📁 Click to select C3D files</div>'
                )

                c3d_files = gr.File(
                    label=None,
                    file_types=[".c3d"],
                    elem_id="file-upload",
                    file_count="multiple",
                )

                c3d_files.change(
                    lambda files: update_display(files, "📁 Click to select C3D files"),
                    [c3d_files],
                    upload_c3d_display,
                )
        run_btn_c3d = gr.Button("3. 🚀 Run Retargeting", variant="primary")

    with gr.Tab("🎥 Video-Based Motion Retargeting"):
        gr.Markdown(
            """
            Extract 3D pose from video and retarget it to a biomechanical model using [Kinesis](https://myolab.ai/blog/myokinesis).

            ⚠️ **Important:** Using Metrabs for video-based motion retargeting which is **ONLY FOR RESEARCH/ACADEMIC USE**.
            Please cite the [paper](https://arxiv.org/abs/2409.06042) if you use this feature.
            For commercial applications, please contact MyoLab.
            """
        )
        video_file = gr.Video(
            label="1. Upload a Video File (Supported formats: MP4, AVI, MOV, MKV)",
            height=400,
            value=os.path.join(
                os.path.dirname(__file__), "../data/13710671_1080_1920_25fps.mp4"
            ),
        )
        run_v2m_btn_video = gr.Button(
            "2. 🚀 Run Retargeting from Video", variant="primary"
        )

    output_file = gr.File(
        label="📥 Download Results - Download the retargeted motion data as a NumPy (.npy) file containing joint angles and metadata.",
        visible=False,
    )
    df_state = gr.State()
    joint_dropdown = gr.Dropdown(
        label="Select Joint Angle(s) to Visualize",
        interactive=False,
        multiselect=True,
        visible=True,
        info="After processing completes, select one or more joint angles to plot. The dropdown will be populated with available joints from the retargeted data.",
    )
    plot_area = gr.Plot(
        label="Joint Angle Visualization - Interactive plot showing the selected joint angles over time. Use the legend to toggle individual joints on/off.",
        visible=False,
    )
    status_box = gr.Textbox(
        label="Processing Status",
        lines=12,
        info="Real-time status updates showing the progress of file uploads, retargeting jobs, and data processing.",
    )

    joint_dropdown.change(
        fn=update_plot,
        inputs=[df_state, joint_dropdown],
        outputs=[plot_area],
    )
    run_btn_c3d.click(
        fn=run_retargeting_c3d,
        inputs=[api_key, c3d_files, markerset],
        outputs=[status_box, output_file, df_state, joint_dropdown, plot_area],
    )
    run_v2m_btn_video.click(
        fn=run_retargeting_video,
        inputs=[api_key, video_file],
        outputs=[
            status_box,
            output_file,
            df_state,
            joint_dropdown,
            plot_area,
            video_file,
        ],
    )

if __name__ == "__main__":
    app.launch(
        share=True,
        # server_port=7860,
    )
