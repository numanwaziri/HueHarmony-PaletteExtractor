import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import colorsys
import pyperclip
def sort_colors(colors):
    # Convert colors to HSL and sort based on hue
    sorted_colors = sorted(colors,
                           key=lambda c: colorsys.rgb_to_hls(*tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)))[0])
    return sorted_colors


def rgb_to_hex(red, green, blue):
    # Convert each RGB component to hexadecimal
    hex_r = format(red, '02x')
    hex_g = format(green, '02x')
    hex_b = format(blue, '02x')

    # Concatenate the hexadecimal values
    hex_code = '#' + hex_r + hex_g + hex_b

    return hex_code.upper()


@st.cache_data
def get_data(img):
    scaled_down = img.copy()
    scaled_down.thumbnail((150, 150))

    df = np.array(scaled_down).reshape(-1, 3)
    return df


def get_clusters(data, model, n_clusters):
    if model == "KMeans":
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        cluster_centers = kmeans.cluster_centers_.astype(int)
        labels = cluster_centers[kmeans.labels_]

    if model == "Gaussian Mixtures":
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(data)
        cluster_centers = gmm.means_.astype(int)
        labels = cluster_centers[gmm.predict(data)]

    return cluster_centers, labels


def stacked_bar_chart(colors):
    # Create evenly spaced values for each color
    values = [1 / len(colors)] * len(colors)

    # Create the trace for each color
    traces = []
    for i in range(len(colors)):
        trace = go.Bar(
            x=[1],
            y=[1],
            width=[values[i]],
            marker=dict(color=colors[i]),
            hovertemplate=f"Hex: <b>{colors[i]}</b><extra></extra>",

            showlegend=False
        )
        traces.append(trace)

    # Create the layout
    layout = go.Layout(

        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.59, 1.4]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Display the figure in Streamlit
    return fig


def create_scatter_plot(data, colors):
    red, green, blue = data.T
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(layout=layout, data=go.Scatter3d(
        x=red,
        y=green,
        z=blue,
        mode='markers',
        text=[f'Hex: {rgb_to_hex(*rgb)}' for rgb in colors],
        marker=dict(
            size=4,
            color=['rgb({},{},{})'.format(*i) for i in colors],
        )
    ))

    fig.update_layout(scene=dict(
        xaxis=dict(zeroline=False, showbackground=False, range=[0, 255]),
        yaxis=dict(zeroline=False, showbackground=False, range=[0, 255]),
        zaxis=dict(zeroline=False, showbackground=False, range=[0, 255]),

        xaxis_title='Red',
        yaxis_title='Green',
        zaxis_title='Blue',
        aspectmode="cube",
        aspectratio=dict(x=1, y=1, z=0.95)
    ), paper_bgcolor='rgba(0,0,0,0)', template="none")

    camera = dict(
        eye=dict(x=2, y=-1, z=0.3)
    )
    fig.update_layout(scene_camera=camera)

    return fig


def main():

    img = None

    st.markdown(
        """
        <style>
        .rounded-heading {
            text-align: center;
            background-color: #ff6f61;
            color: white;
            padding: 13px;
            border-radius: 7px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='rounded-heading'>HueHarmony</h1>", unsafe_allow_html=True)






    st.markdown("<h3 style='text-align: center; color: #ff6f61; text-shadow: 0px 2px 5px rgba(0, 0, 0, 0.07);'>🎨 Image Palette Extractor</h3>", unsafe_allow_html=True)


    with st.expander("⚙️ Options", expanded=False):
        col1, col2 = st.columns(2)
        palette_size = int(col1.number_input("Palette Size", min_value=1, max_value=20, value=6, step=1,
                                             help="Number of colors to extract from the Image"))
        model_name = col2.selectbox("Machine Learning Model", ["KMeans", "Gaussian Mixtures"],
                                    help="Machine Learning model to use for Clustering the pixel RGB values")

    upload_tab, url_tab = st.tabs(["Upload", "Image URL"])
    with upload_tab:

        img_file = st.file_uploader("Upload an image", key="file_uploader", type=["jpg", "jpeg", "png"])
        if img_file is not None:
            img = Image.open(img_file).convert("RGB")

        if st.session_state.get("image_url"):
            st.warning("To use the file uploader, remove the image URL first.")

    with url_tab:

        url = st.text_input("Image URL", key="image_url")
        if url != "":
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            except:
                st.error("The URL is not valid.")

    if img is not None:

        st.image(img, use_column_width=True)




        data = get_data(img)
        clusters, colors = get_clusters(data, model_name, palette_size)

        sorted_colors = sort_colors([rgb_to_hex(*i) for i in clusters])
        st.plotly_chart(stacked_bar_chart(sorted_colors), use_container_width=True)




        if st.button("Copy to Clipboard",use_container_width=True):
            pyperclip.copy(", ".join(sorted_colors))
            st.success("Copied to clipboard!")




        with st.expander("🤖  Details for Machine Learning Enthusiasts ", expanded=True):


            visualization, code = st.tabs(["Visualization", "Code"])

            with code:
                st.write("Implemntation of KMeans Clustering Algorithm from Scratch in Python")
                st.code(f"""
                class KMeans:
    \"""
    K-means clustering algorithm.

    Parameters:
        n_clusters (int): The number of clusters to form.
        max_iterations (int, optional): The maximum number of iterations to perform. Defaults to 100.
        tolerance (float, optional): The tolerance for convergence. Defaults to 1e-4.

    Attributes:
        centroids (ndarray): The final centroids after training.
        labels (ndarray): The labels assigned to each data point after training.

    Methods:
        fit(data): Fit the K-means model to the input data.
        predict(data): Predict the labels for new data points.

    \"""

    def __init__(self, n_clusters, max_iterations=100, tolerance=1e-4):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("The number of clusters (n_clusters) must be a positive integer.")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("The maximum number of iterations (max_iterations) must be a positive integer.")
        if not isinstance(tolerance, float) or tolerance <= 0:
            raise ValueError("The tolerance (tolerance) must be a positive float.")

        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit(self, data):
        \"""
        Fit the K-means model to the input data.

        Parameters:
            data (array-like): The input data.

        \"""
        data = np.array(data)

        if data.ndim != 2:
            raise ValueError("The input data must be two-dimensional.")

        np.random.seed(42)
        centroids = data[np.random.choice(len(data), self.n_clusters, replace=False)]

        for _ in range(self.max_iterations):
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            max_change = np.max(np.abs(centroids - new_centroids))

            if max_change < self.tolerance:
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

    def predict(self, data):
        \"""
        Predict the labels for new data points.

        Parameters:
            data (array-like): The new data points to predict labels for.

        Returns:
            ndarray: The predicted labels for the new data points.

        \"""
        data = np.array(data)

        if data.ndim != 2:
            raise ValueError("The input data must be two-dimensional.")

        distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels
                    """
                        )
            with visualization:
                st.markdown("Explore the color space for the image and ML clustering using <span style='color: #ff6f61;'>**interactive**</span> plots below",
                            unsafe_allow_html=True
                            )
                col1, col2 = st.columns(2)

                col1.markdown("<h6 style='text-align: center;'>Color Space</h6>",
                            unsafe_allow_html=True)

                # col1.write("Color Space")
                col1.plotly_chart(create_scatter_plot(data, colors=data),
                                  use_container_width=True)

                # col2.write(f"Clustering with {model_name} Model")

                col2.markdown(f"<h6 style='text-align: center;'>Clustering with {model_name} Model</h6>",
                              unsafe_allow_html=True)
                col2.plotly_chart(create_scatter_plot(data, colors=colors), use_container_width=True)


if __name__ == '__main__':
    main()
