import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_latent_space():

    with open("master_emotion_anchors.json", "r") as f:
        anchors = json.load(f)

    emotions = list(anchors.keys())
    vectors = np.array(list(anchors.values()))


    print("Flattening 64-D space to 2-D using PCA...")
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)


    plt.figure(figsize=(10, 8))
    plt.style.use('dark_background')


    colors = plt.cm.rainbow(np.linspace(0, 1, len(emotions)))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, emotion in enumerate(emotions):
        x, y, z = vectors_3d[i]
        ax.scatter(x, y, z, color=colors[i], s=200, label=emotion, edgecolors='white')
        ax.text(x, y, z, emotion, fontsize=12)

    plt.title("BEAT Dataset: 64-D Emotion Latent Space (PCA Projection)", fontsize=14)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    plt.grid(True, alpha=0.2)


    plt.savefig("Latent_Space_Map.png", dpi=300, bbox_inches='tight')
    print("SUCCESS: Saved 'Latent_Space_Map.png'")
    plt.show()

if __name__ == "__main__":
    visualize_latent_space()
