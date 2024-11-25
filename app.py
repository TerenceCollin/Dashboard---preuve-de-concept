import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Dashboard : Preuve de concept", page_icon="📝")


@st.cache_data
def load_data():
    # Charger des données ou des modèles qui prennent du temps
    df_train = pd.read_csv("train_final.csv", index_col=None)
    df_train['objects'] = df_train['objects'].apply(ast.literal_eval)
    df_unet_training_metrics = pd.read_csv("unet_training_metrics.csv", index_col=None)
    df_fastscnn_training_metrics = pd.read_csv("fastscnn_training_metrics.csv", index_col=None)
    df_unet_iou_results = pd.read_csv("unet_iou_results.csv", index_col=None)
    df_fastscnn_iou_results = pd.read_csv("fastscnn_iou_results.csv", index_col=None)
    return df_train, df_unet_training_metrics, df_fastscnn_training_metrics, df_unet_iou_results, df_fastscnn_iou_results


df_train, df_unet_training_metrics, df_fastscnn_training_metrics, df_unet_iou_results, df_fastscnn_iou_results = load_data()

# Applatir la liste de dictionnaires et extraire les labels
all_labels = []
for obj_list in df_train['objects']:
    for obj in obj_list:
        all_labels.append(obj['label'])


# Compter le nombre total d'occurrences des labels
label_counts = pd.Series(all_labels).value_counts().reset_index()
label_counts.columns = ['Label', 'Occurrences']

# Calculer le nombre d'occurrences des labels par ligne
label_counts_per_line = []
for obj_list in df_train['objects']:
    label_counts_per_line.append(len(obj_list))


st.title("Réalisez une preuve de concept: Fast-SCNN, évolution pour les systèmes embarqués")

st.write("Sur cette page vous pouvez passer en revue des des informations utiles sur les différentes phases du projet "
         "dans lequel nous avons décortiqué le modèle de segmentation sémantique Fast-SCNN et l'avons mis "
         "à l'épreuve face à notre ancien modèle U-net.")

st.header("Visualisations des données")

image_path_exemple = 'static/exemple.png'

image1 = Image.open(image_path_exemple)

st.subheader("Les images")
st.image(image1, caption="photo et masks vérité terrain de caméra embarquée présent dans le Cityscapes-dataset",  use_container_width=True)

st.write("Les nouveaux modèles ont été entrainé sur 1980 photos de caméras embarqué dans les rues de villes allemandes et "
         "francaises, dont 330 pour la validation. Le but, effectuer de la segmentation sur 8 classes (flat, vehicle, human,"
         " sky, nature, object, construction et rectification).")

# Fonction pour le graphique
def plot_interactive_bar():
    fig = px.bar(label_counts, x='Label', y='Occurrences',
                 title='Distribution des occurrences des Labels',
                 labels={'Label': 'Label', 'Occurrences': 'Nombre d\'occurrences'},
                 color='Occurrences', color_continuous_scale='Viridis')
    fig.update_layout(
        xaxis=dict(
            title="Label",
            tickfont=dict(color="black"),
            titlefont=dict(color="black")
        ),
        yaxis=dict(
            title="Nombre d'occurrences",
            tickfont=dict(color="black"),
            titlefont=dict(color="black"),
            gridcolor='rgba(0, 0, 0, 0.6)'
        ),
        legend=dict(
            font=dict(color="black")
        ),
        title=dict(
            font=dict(color="black")
        )
    )
    st.plotly_chart(fig)


# Fonction pour le box plot interactif
def plot_interactive_box():
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=label_counts_per_line,
        boxmean='sd',
        marker_color='skyblue',
        line_color='black'
    ))
    fig.update_layout(
        title="Box Plot des apparitions de label par ligne",
        xaxis_title="Nombre de Labels",
        yaxis_title="Lignes",
    xaxis=dict(
        title="Nombre de labels",
        tickfont=dict(color="black"),
        titlefont=dict(color="black")
    ),
    yaxis=dict(
        title="Photos",
        tickfont=dict(color="black"),
        titlefont=dict(color="black")
    ),
    legend=dict(font=dict(color="black"))

    )
    st.plotly_chart(fig)

option = st.selectbox(
    "Choisissez le graphique à afficher",
    ("Distribution des Labels", "Distribution des Labels par Ligne")
)

if option == "Distribution des Labels":
    plot_interactive_bar()
elif option == "Distribution des Labels par Ligne":
    plot_interactive_box()

st.write("Comprendre les données d'entrainement et surtout à quoi ressemble notre target peut nous aider à mieux"
         " comprendre les résultats obtenus et nous permettre par la suite de le réequilibrer, si nécessaire, pour améliorer"
         " nos modèles.")

st.write("-" * 50)

st.header("Entrainement")

st.write("Nous avons entrainé le nouveau modèle sur 30 epoch avec calcul du mean IoU et early-stopping."
         " Meme procédure pour l'ancien modèle U-net.")

# Sélecteur de dataset pour le graphique des courbes
dataset_choice = st.selectbox("Choisissez le dataset", ["U-net", "Fast-SCNN"])

# Sélection du DataFrame en fonction du choix
df = df_unet_training_metrics if dataset_choice == "U-net" else df_fastscnn_training_metrics

option = st.selectbox(
    "Choisissez une paire de métriques à afficher",
    [("accuracy", "val_accuracy"), ("loss", "val_loss"), ("mean_iou", "val_mean_iou")]
)


col1, col2 = option

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index, y=df[col1],
    mode='lines+markers',
    name=col1,
    marker_symbol="circle",
    marker=dict(size=8)
))

fig.add_trace(go.Scatter(
    x=df.index, y=df[col2],
    mode='lines+markers',
    name=col2,
    marker_symbol="x",
    marker=dict(size=8, color='rgb(255, 165, 0)'),
    line=dict(color='rgb(255, 165, 0)')
))

# Titre et personnalisation
fig.update_layout(
    title=f"Comparaison de {col1} et {col2} ({dataset_choice})",
    xaxis=dict(
        title="Index",
        tickfont=dict(color="black"),
        titlefont=dict(color="black")
    ),
    yaxis=dict(
        title="Valeur",
        tickfont=dict(color="black"),
        titlefont=dict(color="black"),
        gridcolor='rgba(0, 0, 0, 0.6)'
    ),
    legend=dict(font=dict(color="black"))
)

# Afficher le graphique
st.plotly_chart(fig)

st.write("Grace à ces line plot nous pouvons évaluer la qualité de nos entrainement, repérer les epochs ou les métriques"
         " sont le mieux optimisées, éviter l'over-fitting en monitorant la validation loss...")

st.write("-" * 50)


st.header("Comparaison des prédictions")

st.write("Un masque est généré pour 8 différentes classes, l'intersection over union est ensuite calculée"
         "pour chaque masque prédit, en fonction du masque de vérité terrain.")

# Sélecteur
image_choice = st.selectbox("Choisissez une image", df_unet_iou_results["image_id"])

# Filtrer les données en fonction de l'image sélectionnée
df3_filtered = df_unet_iou_results[df_unet_iou_results["image_id"] == image_choice].drop(columns=["image_id"]).melt(var_name="Feature", value_name="Value")
df4_filtered = df_fastscnn_iou_results[df_fastscnn_iou_results["image_id"] == image_choice].drop(columns=["image_id"]).melt(var_name="Feature", value_name="Value")

# Fusionner les deux DataFrames sur la feature
merged_df = pd.merge(df3_filtered, df4_filtered, on="Feature", suffixes=("_U-net", "_Fast-SCNN"))

# Créer un graphique à barres groupées
fig_bar = go.Figure()

fig_bar.add_trace(go.Bar(
    x=merged_df["Feature"],
    y=merged_df["Value_U-net"],
    name="U-net",
    marker_color='rgb(55, 83, 109)'
))

fig_bar.add_trace(go.Bar(
    x=merged_df["Feature"],
    y=merged_df["Value_Fast-SCNN"],
    name="Fast-SCNN",
    marker_color='rgb(255, 165, 0)'
))

fig_bar.update_layout(
    barmode='group',
    title=f"Comparaison des métriques pour {image_choice}",
    xaxis_title="Feature",
    yaxis_title="IoU",
    legend_title="Dataset",
    xaxis_tickangle=-45,
    xaxis=dict(
        title="Index",
        tickfont=dict(color="black"),
        titlefont=dict(color="black")
    ),
    yaxis=dict(
        title="Valeur",
        tickfont=dict(color="black"),
        titlefont=dict(color="black")
    ),
    legend=dict(font=dict(color="black"))
)

# Afficher le graphique à barres dans Streamlit
st.plotly_chart(fig_bar)

# Afficher les images
image_path_original = os.path.join('static/original_image', f'{image_choice}')
image_path_masks_unet = os.path.join('static/masks_unet', f'{image_choice}')
image_path_masks_fastscnn = os.path.join('static/masks_fastscnn', f'{image_choice}')

image1 = Image.open(image_path_original)

st.subheader("Originale")
st.image(image1, caption="Photo de caméra embarquée", width=300)



st.subheader("U-net segmentation")
image1 = Image.open(image_path_masks_unet)
st.image(image1, caption="Mask généré", width=300)


st.subheader("Fast-SCNN segmentation")
image2 = Image.open(image_path_masks_fastscnn)
st.image(image2, caption="Mask généré", width=300)

st.write("-" * 50)

st.header("Conclusion")

st.write("En comparant visuellement les masques ainsi que les résultats IoU, on démontre que le modèle "
         "Fast-SCNN, avec des mean IoU tournant autour de 0.6, est une réelle amélioration par rapport à notre "
         "ancien modèle U-Net, dont les mean IoU oscillent entre 0.2 et 0.4.")
st.write("De plus le modèle Fast-SCNN a été pensé avec en tete les systèmes embarqué, ce qui correspond exactement"
         " à notre problèmatique métier. Le choix de Fast-SCNN est donc validé !")
