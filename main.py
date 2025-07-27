import math
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import colorsys
from streamlit_lottie import st_lottie
import json
st.set_page_config(
    page_title="HueHarmony",
    page_icon="🎨",
)
def step (r,g,b, repetitions=1):
    lum = math.sqrt( .241 * r + .691 * g + .068 * b )
    h, s, v = colorsys.rgb_to_hsv(r,g,b)
    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)
    if h2 % 2 == 1:
        v2 = repetitions - v2
        lum = repetitions - lum
    return (h2, lum, v2)
def sort_colors(colors):
    # Convert hex colors to RGB and sort based on the step function
    sorted_colors = sorted(colors, key=lambda c: step(*tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)),repetitions=8))
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


def create_scatter_plot(data, colors,title):
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
    fig.update_layout(scene_camera=camera
                      )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(
                family='Arial',
                size=14,
                color="grey"
            ),
            xref='paper',
            yref='paper',
        ),
        title_x=0.5,
        title_y=0.97
    )

    return fig


def main():
    img = None


    st.markdown(
        """
        <style>
        .rounded-heading {
            text-align: center;
            background: linear-gradient(to right, #ff6f61, #e23e57);
            color: white;
            padding: 23px;
            border-radius: 7px;
            box-shadow: 0px 3px 4px rgba(0, 0, 0, 0.2);
        }
        
        .rounded-heading:hover {
            background: linear-gradient(to left, #ff6f61, #e23e57);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown("<h1 class='rounded-heading' style='color: white'>HueHarmony</h1>", unsafe_allow_html=True)

    st.markdown("""
        <div style="
            text-align: center;
            color: #ff6f61;
            font-size: 28px;
            font-weight: 600;
            padding: 10px 0;
            margin-bottom: 15px;
            border-bottom: 2px solid #ff6f61;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.08);
        ">
            🎨 Image Palette Extractor<br><span style="font-size: 18px; font-weight: 400; color:grey;">by Ahmad Noman Waziri</span>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Options", expanded=True):
        col1, col2 = st.columns(2)
        palette_size = int(col1.number_input("Palette Size", min_value=1, max_value=24, value=8, step=1,
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

    if img is None:

        path = "./Assets/lotte.json"
        with open(path, "r") as file:
            url = json.load(file)

        st_lottie(url,
                  reverse=True,
                  height=360,
                  speed=1,
                  loop=True,
                  quality='high',
                  key='colors'
                  )

    if img is not None:
        if img.size[1]>200:

            height = 200  # Set your desired height in pixels

            # Resize the image while maintaining the aspect ratio
            width_percent = (height / float(img.size[1]))
            width_size = int((float(img.size[0]) * float(width_percent)))
            resized_image = img.resize((width_size, height), Image.Resampling.LANCZOS)

        else:
            resized_image = img


        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(resized_image)



        data = get_data(img)
        clusters, colors = get_clusters(data, model_name, palette_size)
        sorted_colors = sort_colors([rgb_to_hex(*i) for i in clusters])
        st.plotly_chart(stacked_bar_chart(sorted_colors), use_container_width=True)

        st.download_button('Download Palette', "\n".join(sorted_colors),use_container_width=True,file_name='colors.txt',on_click ='ignore')


        with st.expander("🤖  Details for Machine Learning Enthusiasts ", expanded=True):


            visualization, code = st.tabs(["Visualization", "Explaination"])


            with visualization:
                st.markdown(f"Discover how the {model_name} algorithm clusters colors in RGB space through the <span style='color: #ff6f61;'>**interactive**</span> plots below.",
                            unsafe_allow_html=True
                            )
                col1, col2 = st.columns(2)

                col1.plotly_chart(create_scatter_plot(data, colors=data,title = "Original RGB Color Space"),
                                  use_container_width=True)

                col2.plotly_chart(create_scatter_plot(data, colors=colors, title = f'Clustering with {model_name} Model'), use_container_width=True)

                with code:
                    st.markdown(f"# How {model_name} Picks the Palette Colors")
                    if model_name == "KMeans":
                        st.markdown("""
                    Every image is made of thousands of tiny dots called **pixels**, and each pixel has a color made from red, green, and blue (RGB) values.

                    The **KMeans algorithm** helps reduce all those colors into a smaller, more meaningful set, like a color palette. Here’s how it works:

                    **1. Pick starting colors**  
                    It randomly selects *n* colors from your image as starting points (called “cluster centers”). You choose how many (e.g. 5 colors).

                    **2. Group similar colors together**  
                    Each pixel is assigned to the closest color center based on how similar their RGB values are.

                    **3. Update the colors**  
                    For each group, it calculates the average color of the pixels assigned to that group. That becomes the new center.

                    **4. Repeat until stable**  
                    Steps 2 and 3 repeat until the colors stop changing much. This means the groups are stable.

                    ---

                    ✅ The final group centers become your image’s **main colors**, giving you a clean, simplified color palette.
                    """)
                    else:
                        st.markdown("""
                        Your image is made up of tiny dots called **pixels**, and each pixel has a color made from red, green, and blue (RGB) values.

                        GMM helps find the main colors by treating pixel colors as data points and grouping them based on patterns. Here's how it works:

                        **1. Guess some color groups**  
                        GMM starts by assuming n color groups. Each group is shaped like a soft blob (called a "Gaussian") that can stretch in different directions.

                        **2. Measure how likely each pixel fits in each group**  
                        Instead of assigning pixels to just one group, GMM calculates how likely each pixel belongs to each color group.

                        **3. Adjust the color groups**  
                        Based on those likelihoods, GMM updates the position, shape, and size of each group so they better match the pixel colors.

                        **4. Repeat until it fits well**  
                        This process is repeated, measuring likelihoods and updating groups, until the model finds a good fit for all the pixels.

                        ✅ The result is a set of smooth color groups that represent the image's main colors, even when colors are blended or gradual.
                        """)


if __name__ == '__main__':
    main()
