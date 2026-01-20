"""
Copyright (c) 2026 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.

MyoSDK Kinesis App
"""

import os
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objs as go
import spaces
import torch
from metrabs_pytorch.scripts.run_video import run_metrabs_video
from myo_tools.mjs.marker.marker_api import get_marker_names
from myo_tools.utils.file_ops.dataframe_utils import from_array_to_dataframe
from myo_tools.utils.file_ops.io_utils import from_qpos_to_joint_angles
from myo_tools.utils.mocap_ops.mocap_utils import rotate_mocap_ydown_to_zup
from myosdk import Client

PLOT_CONFIG = {
    "plot_bgcolor": "#0f172a",
    "paper_bgcolor": "#0f172a",
    "font": {"color": "#e2e8f0", "family": "Inter, system-ui, sans-serif"},
    "xaxis": {"gridcolor": "#1e293b", "linecolor": "#334155"},
    "yaxis": {"gridcolor": "#1e293b", "linecolor": "#334155"},
}


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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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


def load_all_videos():
    video_dir = os.path.join(os.path.dirname(__file__), "../data")
    return [
        os.path.abspath(os.path.join(video_dir, f))
        for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]


def create_info_text(title, info_text):
    return f"""
    <div>
        <style>
        .info-icon {{
            position: relative;
            display: inline-block;
            cursor: help;
            color: #6b7280;
            margin-left: 5px;
            font-size: 0.9em;
        }}

        .info-icon .tooltip-text {{
            visibility: hidden;
            position: absolute;
            width: 250px;
            background-color: #1f2937;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            left: 0;
            top: 25px;
            z-index: 9999;
            font-size: 0.9em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            opacity: 0;
            transition: opacity 0.2s, visibility 0.2s;
        }}

        .info-icon:hover .tooltip-text,
        .tooltip-text:hover {{
            visibility: visible;
            opacity: 1;
        }}

        .tooltip-text a {{
            color: #60a5fa;
            text-decoration: none;
        }}

        .tooltip-text a:hover {{
            text-decoration: underline;
        }}
        </style>

        <div style="margin-bottom: 10px;">
            <strong>{title}</strong>
            <span class="info-icon">
                ℹ️
                <span class="tooltip-text">
                    {info_text}
                </span>
            </span>
        </div>
    </div>
    """


# ------------------------------------------------------------
# Retargeting
# ------------------------------------------------------------
def run_retargeting_c3d(api_key, c3d_files, markerset_file):
    status = []
    output_files = []
    output_glb_files = []

    # Initial validation
    if not api_key:
        api_key = os.getenv("MYOSDK_API_KEY")
        if not api_key:
            gr.Warning("❌ Error: API key is missing!", duration=5)
            yield (
                "❌ Error: API key is missing or invalid",
                None,
                None,
                gr.update(value=[], visible=True),
                gr.update(visible=False),
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
            gr.update(visible=False),
        )

    try:
        # Initialize client
        status.append("🔹 Initializing MyoSDK client...")
        init_time = time.time()
        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=False
        ), gr.update(visible=False)
        client = Client(api_key=api_key)

        status.append(
            f"🔹 MyoSDK client initialized in { time.time() - init_time:.2f} seconds"
        )
        init_time = time.time()
        # Upload markerset
        status.append("🔹 Uploading markerset file...")
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        ), gr.update(visible=False)
        mk_asset = client.assets.upload_file(markerset_file.name)

        status.append(
            f"🔹 Markerset file uploaded in {time.time() - init_time:.2f} seconds"
        )
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        ), gr.update(visible=False)

        mk_id = mk_asset["asset_id"]

        # Process each C3D file
        total_files = len(c3d_files)
        for idx, f in enumerate(c3d_files):
            status.append(
                f"➡ Processing file {idx + 1}/{total_files}: {os.path.basename(f)}"
            )

            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            ), gr.update(visible=False)

            init_time = time.time()
            tracker_asset = client.assets.upload_file(f)
            status.append(
                f"\t🔹 C3D file uploaded in {time.time() - init_time:.2f} seconds"
            )
            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            ), gr.update(visible=False)
            init_time = time.time()
            job = client.jobs.start_retarget(
                tracker_asset_id=tracker_asset["asset_id"],
                markerset_asset_id=mk_id,
                export_glb=True,
            )

            status.append(
                f"\t🔹 Retargeting job started in {time.time() - init_time:.2f} seconds ... Processing may take a few seconds depending on the file size."
            )
            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            ), gr.update(visible=False)
            init_time = time.time()
            result = client.jobs.wait(job["job_id"])

            status.append(
                f"\t🔹 Retargeting job completed in {time.time() - init_time:.2f} seconds"
            ), gr.update(visible=False),
            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            ), gr.update(visible=False)
            assert (
                result["status"] == "SUCCEEDED"
            ), f"Failed retarget for {os.path.basename(f)}"

            status.append(f"\t✅ Retargeting completed for {os.path.basename(f)}")
            base = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(tempfile.gettempdir(), base + ".parquet")
            output_glb_path = os.path.join(tempfile.gettempdir(), base + ".glb")

            client.assets.download(
                result["output"]["retarget_output_asset_ids"]["qpos"], out_path
            )

            output_files.append(out_path)
            client.assets.download(
                result["output"]["retarget_output_asset_ids"]["motion"], output_glb_path
            )
            output_glb_files.append(output_glb_path)

        assert os.path.exists(
            output_files[0]
        ), f"Failed to download retargeted data for {os.path.basename(f)}"

        # Load angles from first output file
        status.append("🔹 Loading animation and angle data...")

        assert os.path.getsize(output_glb_files[0]) > 0
        time.sleep(0.1)  # allow filesystem flush
        yield "\n".join(status), None, None, gr.update(
            interactive=True, visible=True
        ), gr.update(visible=True), gr.update(visible=True, value=output_glb_files[0])

        df = from_qpos_to_joint_angles(output_files[0])

        angle_list = list(df.columns[1:])
        initial_value = [angle_list[0]] if angle_list else []

        status.append("✅ Complete!")
        yield (
            "\n".join(status),
            gr.update(value=output_files, visible=True),
            df,
            gr.update(choices=angle_list, value=initial_value, visible=True),
            gr.update(visible=True),
            gr.update(visible=True, value=output_glb_files[0]),
        )

    except Exception as e:
        yield (
            f"❌ {e}",
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


@spaces.GPU
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
            gr.Warning("❌ Error: API key is missing!", duration=5)
            yield (
                "❌ Error: API key is missing or invalid",
                None,
                None,
                gr.update(visible=False),
                gr.update(visible=False),
                video_file,
                gr.update(visible=False),
            )
            return

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
            gr.update(visible=False),
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
        ), video_path, gr.update(visible=False)

        results = list(
            run_metrabs_video(video_path=video_path, device=DEVICE, visualize=False)
        )
        markers = rotate_mocap_ydown_to_zup(
            np.array([res["poses3d"] for res in results]).squeeze()
        )

        fps = (
            results[0]["fps"] if results else 25.0
        )  # Default to 25 fps if not available

        video_with_keypoints = os.path.join(
            tempfile.gettempdir(), "video_with_keypoints.mp4"
        )
        save_video_with_keypoints(results, video_with_keypoints)

        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=True
        ), video_with_keypoints, gr.update(visible=False)
        status.append(
            f"🔹 Pose Extraction from Video Completed in {time.time() - init_time:.2f} seconds with {len(markers)} frames extracted ({((time.time() - init_time)/len(markers)):.2f} seconds per frame)"
        )
        print("🔹 Pose Extraction from Video Completed")
        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=False
        ), video_with_keypoints, gr.update(visible=False)
        # Initialize client
        status.append("🔹 Initializing MyoSDK client...")
        init_time = time.time()
        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=False,
        ), video_with_keypoints, gr.update(visible=False)
        client = Client(api_key=api_key)

        status.append(
            f"🔹 MyoSDK client initialized in { time.time() - init_time:.2f} seconds"
        )
        init_time = time.time()
        # Upload markerset
        status.append("🔹 Uploading markerset file...")
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False,
        ), video_with_keypoints, gr.update(visible=False)

        markerset_file_name = "markersets/movi_metrabs_markerset.xml"

        mk_asset = client.assets.upload_file(markerset_file_name)

        status.append(
            f"🔹 Markerset file uploaded in {time.time() - init_time:.2f} seconds"
        )

        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        ), video_with_keypoints, gr.update(visible=False)
        init_time = time.time()

        marker_names = get_marker_names(markerset_file_name)

        from_array_to_dataframe(
            np.array([res["poses3d"] for res in results]).squeeze(),
            marker_names,
            fps,
            "./video_trackers_original.parquet",
        )

        fn_parquet = os.path.join(tempfile.gettempdir(), "video_trackers.parquet")
        from_array_to_dataframe(markers, marker_names, fps, fn_parquet)
        tracker_asset = client.assets.upload_file(fn_parquet)

        print("fn_parquet: ", fn_parquet)

        init_time = time.time()
        job = client.jobs.start_retarget(
            tracker_asset_id=tracker_asset["asset_id"],
            markerset_asset_id=mk_asset["asset_id"],
            export_glb=True,
        )

        status.append(
            f"\t🔹 Retargeting job started in {time.time() - init_time:.2f} seconds ... Processing may take a few seconds depending on the video length."
        )
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        ), video_with_keypoints, gr.update(visible=False)
        init_time = time.time()
        result = client.jobs.wait(job["job_id"])

        status.append(
            f"\t🔹 Retargeting job completed in {time.time() - init_time:.2f} seconds"
        )
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        ), video_with_keypoints, gr.update(visible=False)

        print("STATUS: ", result["status"])
        assert (
            result["status"] == "SUCCEEDED"
        ), f"Failed retarget for {os.path.basename(video_path)}"

        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(tempfile.gettempdir(), base + ".parquet")
        output_glb_path = os.path.join(tempfile.gettempdir(), base + ".glb")
        client.assets.download(
            result["output"]["retarget_output_asset_ids"]["qpos"], out_path
        )

        assert os.path.exists(
            out_path
        ), f"Failed to download retargeted data for {os.path.basename(video_path)}"

        client.assets.download(
            result["output"]["retarget_output_asset_ids"]["motion"], output_glb_path
        )

        # Load angles from first output file
        status.append("🔹 Loading animation and angle data...")
        yield "\n".join(status), None, None, gr.update(
            interactive=True, visible=True
        ), gr.update(visible=True), video_with_keypoints, gr.update(
            visible=True, value=output_glb_path
        )

        df = from_qpos_to_joint_angles(out_path)

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
            gr.update(visible=True, value=output_glb_path),
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
            fig.add_trace(go.Scatter(x=df["time"], y=df[j], mode="lines", name=j))

    fig.update_layout(
        title="Joint Angles",
        xaxis_title="Time (s)",
        yaxis_title="Angle Value (degrees)",
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(color="#F0F0F0", family="Arial"),
        xaxis=dict(gridcolor="#444444", linecolor="#F0F0F0", tickcolor="#F0F0F0"),
        yaxis=dict(gridcolor="#444444", linecolor="#F0F0F0", tickcolor="#F0F0F0"),
        legend=dict(font=dict(color="#F0F0F0")),
    )
    return fig


with gr.Blocks() as app:

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                """
                ## MyoSDK Kinesis
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
                gr.HTML(
                    create_info_text(
                        "1. Upload a Markerset File",
                        "Upload an XML file that defines the marker set configuration. This file specifies which markers are used and their anatomical locations."
                        + 'See <a href="https://markerset-editor.myolab.ai">Markerset Editor</a> for more details.',
                    )
                )

                markerset = gr.File(
                    # label=None,
                    file_types=[".xml"],
                    elem_id="file-upload-markerset",
                    value=os.path.join(
                        os.path.dirname(__file__), "../markersets/cmu_markerset.xml"
                    ),
                )

            with gr.Column(scale=2):
                gr.HTML(
                    create_info_text(
                        "2. Upload C3D Motion Capture File(s)",
                        "Upload one or more C3D files containing 3D marker trajectories from motion capture systems."
                        + 'Example from CMU dataset: <a href="https://mocap.cs.cmu.edu/subjects/35/35_30.c3d">35_30.c3d</a>',
                    )
                )
                c3d_files = gr.File(
                    label=None,
                    file_types=[".c3d"],
                    elem_id="file-upload-c3d",
                    file_count="multiple",
                    value=[
                        os.path.join(os.path.dirname(__file__), "../data/35_30.c3d")
                    ],
                )

        run_btn_c3d = gr.Button("3. 🚀 Run Retargeting", variant="primary")

    with gr.Tab("🎥 Video-Based Motion Retargeting"):
        gr.Markdown(
            """
            Extract 3D pose from video and retarget it to a biomechanical model using [Kinesis](https://myolab.ai/blog/myokinesis).

            ⚠️ **Important:** Using Metrabs for video-based motion retargeting which is **ONLY FOR RESEARCH/ACADEMIC USE**.
            Please cite the [paper](https://arxiv.org/abs/2409.06042) if you use this feature.
            For commercial applications, please contact MyoLab at contacts@myolab.ai.
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

    output_3d_motion = gr.Model3D(
        label="3D Motion Visualization",
        visible=False,
    )
    output_file = gr.File(
        label="📥 Download Results - Download the retargeted motion data as a Parquet (.parquet) file containing joint angles and metadata.",
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
        outputs=[
            status_box,
            output_file,
            df_state,
            joint_dropdown,
            plot_area,
            output_3d_motion,
        ],
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
            output_3d_motion,
        ],
    )

if __name__ == "__main__":
    app.launch(
        share=True,
        # server_port=7860,
    )
