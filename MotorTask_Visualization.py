from matplotlib import pyplot as plt
import streamlit as st
from nilearn import plotting, image, datasets
import nibabel as nib
import plotly.graph_objs as go
import numpy as np
import os
import streamlit.components.v1 as components  # Import components for embedding HTML

def load_fmri_data(file_path):
    """Load fMRI data from a NIfTI file."""
    return image.load_img(file_path)

def visualize_3d(stat_img, threshold):
    """Visualize the fMRI statistical map on a 3D brain model using nilearn."""
    display = plotting.view_img(stat_img, threshold=threshold)
    return display

def visualize_2d(stat_img, threshold, display_mode='ortho'):
    """Visualize the fMRI statistical map on 2D slices."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plotting.plot_stat_map(
        stat_img,
        threshold=threshold,
        display_mode=display_mode,
        cut_coords=(0, -52, 18),
        title="fMRI Activity",
        figure=fig,
        axes=ax,
        annotate=False  # Optional: Disable annotation if not needed
    )
    st.pyplot(fig)
    plt.close(fig)  # Close the figure to free memory

def get_brain_mesh():
    """Get the brain mesh for 3D visualization."""
    # Using 'fsaverage' as the standard mesh
    return datasets.fetch_surf_fsaverage()

def compare_tasks(uploaded_files):
    st.write("### Activity Comparison Across Tasks")
    # For simplicity, we'll compare the number of active voxels above a threshold
    threshold = st.slider("Select Threshold for Comparison (Z-score)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

    task_names = [file.name for file in uploaded_files]
    active_voxels = []

    for uploaded_file in uploaded_files:
        temp_dir = "temp_fmri"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        stat_img = load_fmri_data(file_path)
        data = stat_img.get_fdata()
        count = np.sum(data > threshold)
        active_voxels.append(count)

    # Create a bar chart using Plotly
    fig = go.Figure([go.Bar(x=task_names, y=active_voxels, marker_color='indigo')])
    fig.update_layout(
        title='Number of Active Voxels Above Threshold',
        xaxis_title='Motor Tasks',
        yaxis_title='Active Voxels'
    )
    st.plotly_chart(fig)

def main():
    st.title("fMRI Data Visualization for Motor Tasks")
    st.write("""
    Upload your fMRI data and visualize brain regions activated during different motor tasks.
    """)

    # Sidebar for user inputs
    st.sidebar.header("Upload fMRI Data")
    uploaded_files = st.sidebar.file_uploader(
        "Choose fMRI NIfTI files",
        accept_multiple_files=True,
        type=['.nii', '.nii.gz']
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"Task: {uploaded_file.name}"):
                # Save uploaded file to a temporary directory
                temp_dir = "temp_fmri"
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load fMRI data
                try:
                    stat_img = load_fmri_data(file_path)
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {e}")
                    continue

                # Display basic information
                st.write("### Data Information")
                st.write(f"**File Name:** {uploaded_file.name}")
                st.write(f"**Shape:** {stat_img.shape}")
                st.write(f"**Affine:**\n{stat_img.affine}")

                # Threshold slider
                threshold = st.slider(
                    f"Select Threshold (Z-score) for {uploaded_file.name}",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.1
                )

                # 3D Visualization
                if st.button(f"Show 3D Visualization for {uploaded_file.name}"):
                    with st.spinner('Generating 3D visualization...'):
                        display = visualize_3d(stat_img, threshold)
                        # Embed the interactive 3D visualization using components
                        html_content = display._repr_html_()
                        components.html(html_content, height=600, scrolling=True)

                # 2D Visualization
                st.write("### 2D Slice Visualization")
                display_mode = st.selectbox(
                    f"Select Display Mode for {uploaded_file.name}",
                    options=['ortho', 'x', 'y', 'z'],
                    key=uploaded_file.name
                )
                with st.spinner('Generating 2D visualization...'):
                    visualize_2d(stat_img, threshold, display_mode=display_mode)

        # Comparison Section
        if len(uploaded_files) > 1:
            compare_tasks(uploaded_files)

    else:
        st.info("No files uploaded. Displaying sample fMRI data.")

        # Fetch sample data from nilearn
        dataset = datasets.fetch_neurovault_motor_task()
        sample_file = dataset.images[0]
        stat_img = load_fmri_data(sample_file)

        st.write("### Sample fMRI Data: Motor Task")
        st.write(f"**File Name:** {sample_file}")
        st.write(f"**Shape:** {stat_img.shape}")
        st.write(f"**Affine:**\n{stat_img.affine}")

        # Threshold slider
        threshold = st.slider("Select Threshold (Z-score)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

        # 3D Visualization
        if st.button("Show 3D Visualization for Sample Data"):
            with st.spinner('Generating 3D visualization...'):
                display = visualize_3d(stat_img, threshold)
                html_content = display._repr_html_()
                components.html(html_content, height=600, scrolling=True)

        # 2D Visualization
        st.write("### 2D Slice Visualization")
        display_mode = st.selectbox("Select Display Mode", options=['ortho', 'x', 'y', 'z'], key='sample')
        with st.spinner('Generating 2D visualization...'):
            visualize_2d(stat_img, threshold, display_mode=display_mode)

if __name__ == "__main__":
    main()
