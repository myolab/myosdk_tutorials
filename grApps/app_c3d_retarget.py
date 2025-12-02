"""
MyoSDK Retargeting App with Joint Dropdown Visualization (fixed multi-joint plot)
"""

import os
import tempfile

import gradio as gr
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

    if not api_key:
        api_key = os.getenv("MYOSDK_API_KEY")
        if not api_key:
            return (
                "❌ Error: API key missing",
                None,
                None,
                gr.update(choices=[], value=[]),
                None,
            )

    try:
        client = Client(api_key=api_key, base_url="https://v2m-alb-us-east-1.myolab.ai")
        status.append("🔹 MyoSDK client initialized")

        if markerset_file is None:
            return (
                "❌ Error: Markerset XML file is required",
                None,
                None,
                gr.update(choices=[], value=[]),
                None,
            )

        mk_asset = client.assets.upload_file(markerset_file.name)
        mk_id = mk_asset["asset_id"]

        for f in c3d_files:
            status.append(f"➡ Processing file {os.path.basename(f)}")
            print("\n".join(status))
            c3d_asset = client.assets.upload_file(f)
            job = client.jobs.start_retarget(
                c3d_asset_id=c3d_asset["asset_id"],
                markerset_asset_id=mk_id,
            )
            result = client.jobs.wait(job["job_id"])
            print(f"✅ Retargeting job completed. Status: {result['status']}")
            if result["status"] != "SUCCEEDED":
                status.append("❌ Failed retarget")
                continue

            base = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(tempfile.gettempdir(), base + ".npy")
            client.assets.download(result["output"]["qpos_asset_id"], out_path)
            output_files.append(out_path)

        if not output_files:
            return (
                "\n".join(status),
                None,
                None,
                gr.update(choices=[], value=[]),
                None,
            )

        # Load angles
        data = np.load(output_files[0]).squeeze()
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        num_cols = data.shape[1]
        df = pd.DataFrame(data, columns=[f"angle{i}" for i in range(num_cols)])
        df.insert(0, "frame", df.index)

        angle_list = list(df.columns[1:])  # no "frame"
        first_angle = angle_list[0] if angle_list else None

        # Keep initial plot as None, will be populated by dropdown change
        initial_plot = None

        # For multiselect, value should be a list
        initial_value = [first_angle] if first_angle else []

        return (
            "\n".join(status),
            output_files[0],
            df,
            gr.update(choices=angle_list, value=initial_value),
            initial_plot,
        )

    except Exception as e:
        print("error:", e)
        return f"❌ {e}", None, None, gr.update(choices=[], value=[]), None


# Connect retargeting
def retarget_wrapper(api_key, c3d_files, markerset):
    status, file, df, dropdown_update, _ = run_retargeting(
        api_key, c3d_files, markerset
    )

    if df is not None and not df.empty:
        initial_joints = list(dropdown_update["value"])
        initial_plot = update_plot(df, initial_joints)
    else:
        initial_plot = None

    return status, file, df, dropdown_update, initial_plot


# Update plot when choosing angle
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
        plot_bgcolor="#1E1E1E",  # Matches Gradio dark panel
        paper_bgcolor="#1E1E1E",  # Overall figure background
        font=dict(color="#F0F0F0", family="Arial"),  # Text color
        xaxis=dict(
            title="Frame",
            gridcolor="#444444",  # Slightly lighter than background
            linecolor="#F0F0F0",
            tickcolor="#F0F0F0",
        ),
        yaxis=dict(
            title="Value", gridcolor="#444444", linecolor="#F0F0F0", tickcolor="#F0F0F0"
        ),
        legend=dict(font=dict(color="#F0F0F0")),
    )
    return fig


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=3):  # 75% width
            gr.Markdown(
                """
    ## MyoSDK Retargeting with Joint Visualization
    You need an API key to use this app. Get it on [dev.myolab.ai](dev.myolab.ai). Follow the instructions [here](https://docs.myolab.ai/docs/myosdk/getting-started/api-key)
    """
            )
        with gr.Column(scale=1):
            api_key = gr.Textbox(label="API Key", type="password")

    markerset = gr.File(label="Upload a markerset XML file", file_types=[".xml"])
    c3d_files = gr.File(
        label="C3D Motion Capture File", file_types=[".c3d"], file_count="multiple"
    )

    run_btn = gr.Button("Run Retargeting")

    output_file = gr.File(label="Download")

    df_state = gr.State()

    joint_dropdown = gr.Dropdown(
        label="Select Joint Angle(s)",
        choices=[],
        value=[],
        interactive=True,
        multiselect=True,
    )

    # LinePlot configured for multi-series plotting
    plot_area = gr.Plot(label="Angle Plot")

    status_box = gr.Textbox(label="Status", lines=10)

    run_btn.click(
        fn=retarget_wrapper,
        inputs=[api_key, c3d_files, markerset],
        outputs=[
            status_box,
            output_file,
            df_state,
            joint_dropdown,
            plot_area,
        ],
    )

    joint_dropdown.change(
        fn=update_plot,
        inputs=[df_state, joint_dropdown],
        outputs=[plot_area],
    )


if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
