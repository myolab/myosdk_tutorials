"""
MyoSDK Retargeting App
"""

import os
import tempfile

import gradio as gr
import myosdk
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from myosdk import Client


# ------------------------------------------------------------
# Retargeting
# ------------------------------------------------------------
def run_retargeting(api_key, c3d_files, markerset_file):
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

    if markerset_file is None:
        yield (
            "❌ Error: Markerset XML file is required",
            None,
            None,
            gr.update(value=[], visible=True),
            gr.update(visible=False),
        )

    try:
        # Initialize client
        status.append("🔹 Initializing MyoSDK client...")
        yield "\n".join(status), None, None, gr.update(visible=False), gr.update(
            visible=False
        )
        client = Client(api_key=api_key, base_url="https://v2m-alb-us-east-1.myolab.ai")

        # Upload markerset
        status.append("🔹 Uploading markerset file...")
        yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
            visible=False
        )
        mk_asset = client.assets.upload_file(markerset_file.name)
        mk_id = mk_asset["asset_id"]

        # Process each C3D file
        total_files = len(c3d_files)
        for idx, f in enumerate(c3d_files):
            status.append(
                f"➡ Processing file {idx + 1}/{total_files}: {os.path.basename(f)}"
            )
            progress_value = 0.05 + 0.8 * (idx / total_files)
            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            ), progress_value

            c3d_asset = client.assets.upload_file(f)
            job = client.jobs.start_retarget(
                c3d_asset_id=c3d_asset["asset_id"],
                markerset_asset_id=mk_id,
            )
            result = client.jobs.wait(job["job_id"])

            if result["status"] != "SUCCEEDED":
                status.append(f"❌ Failed retarget for {os.path.basename(f)}")
                continue

            status.append(f"✅ Retargeting completed for {os.path.basename(f)}")
            base = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(tempfile.gettempdir(), base + ".npy")
            client.assets.download(result["output"]["qpos_asset_id"], out_path)
            output_files.append(out_path)

        if not output_files:
            yield "\n".join(status), None, None, gr.update(value=[]), gr.update(
                visible=False
            ), 1.0

        # Load angles from first output file
        status.append("🔹 Loading angle data...")
        yield "\n".join(status), None, None, gr.update(visible=True), gr.update(
            visible=True
        )

        data = np.load(output_files[0])
        joints_qpos = data["joints_qpos"].squeeze()
        joint_names = data["joints_qpos_colnames"]

        df = pd.DataFrame(joints_qpos, columns=[jn for jn in joint_names])
        df.insert(0, "frame", df.index)

        angle_list = list(df.columns[1:])
        initial_value = [angle_list[0]] if angle_list else []

        status.append("✅ Complete!")
        yield (
            "\n".join(status),
            output_files[0],
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
            1.0,
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


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                """
                ## MyoSDK Retargeting with Joint Visualization
                Get your API key at https://dev.myolab.ai
                """
            )
        with gr.Column(scale=1):
            api_key = gr.Textbox(label="API Key", type="password")

    markerset = gr.File(label="Upload a markerset XML file", file_types=[".xml"])
    c3d_files = gr.File(
        label="C3D Motion Capture File", file_types=[".c3d"], file_count="multiple"
    )

    run_btn = gr.Button("Run Retargeting")

    status_box = gr.Textbox(label="Status", lines=12)
    output_file = gr.File(label="Download", visible=False)
    df_state = gr.State()
    joint_dropdown = gr.Dropdown(
        label="Select Joint Angle(s)",
        interactive=True,
        multiselect=True,
        visible=True,
    )
    plot_area = gr.Plot(label="Angle Plot", visible=False)

    gr.Markdown(f"MyoSDK version {myosdk.__version__}")

    run_btn.click(
        fn=run_retargeting,
        inputs=[api_key, c3d_files, markerset],
        outputs=[status_box, output_file, df_state, joint_dropdown, plot_area],
    )

    joint_dropdown.change(
        fn=update_plot,
        inputs=[df_state, joint_dropdown],
        outputs=[plot_area],
    )

if __name__ == "__main__":
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
    )
