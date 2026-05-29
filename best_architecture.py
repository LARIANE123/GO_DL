# --- Imports ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np, gc, golois
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)

# ==========================================================
# 1. Fonctions d'augmentation de données (symétries du plateau)
# ==========================================================
def _apply_symmetry_single(board, policy):
    """
    Applique une symétrie aléatoire sur un plateau :
    - rotation (0°, 90°, 180°, 270°)
    - flip horizontal optionnel
    La même transformation est appliquée à la policy.
    """
    k, f = np.random.randint(0, 4), np.random.randint(0, 2)

    # Rotation
    b = np.rot90(board, k, axes=(0, 1))
    p = np.rot90(policy.reshape(19, 19), k)

    # Flip horizontal
    if f:
        b, p = np.flip(b, 1), np.flip(p, 1)

    # Remise au format attendu
    return b.astype("float32"), p.reshape(361).astype("float32")


def augment_batch_by_symmetry(X, P):
    """
    Applique une augmentation par symétrie
    indépendamment à chaque élément du batch.
    """
    N = X.shape[0]
    Xn, Pn = np.empty_like(X), np.empty_like(P)

    for i in range(N):
        Xn[i], Pn[i] = _apply_symmetry_single(X[i], P[i])

    return Xn, Pn

# ==========================================================
# 2. Hyperparamètres
# ==========================================================
planes  = 31      # Plans d'entrée (features du plateau)
moves   = 361     # Nombre de coups possibles (19x19)
N       = 10000   # Taille des buffers
epochs  = 10000    # Nombre total d'époques
batch   = 128     # Taille des batchs
filters = 47      # Canaux convolutifs
l2      = regularizers.l2(1e-4)  # Régularisation L2

# ==========================================================
# 3. Définition du modèle
# ==========================================================
def build_model():

    # ------------------------------------------------------
    # Bloc ConvNeXt :
    # depthwise 7x7 + bottleneck pointwise + GELU
    # ------------------------------------------------------
    def ConvNeXtBlock(x, expansion=2):
        c = x.shape[-1]
        y = layers.DepthwiseConv2D(7, padding="same")(x)
        y = layers.Conv2D(c * expansion, 1)(y)
        y = layers.Activation("gelu")(y)
        y = layers.Conv2D(c, 1)(y)
        return layers.Add()([x, y])  # skip connection

    # ------------------------------------------------------
    # Local Channel Attention (léger)
    # ------------------------------------------------------
    def LCA(x, r=6):
        c = x.shape[-1]
        g = layers.GlobalAveragePooling2D()(x)
        g = layers.Dense(c // r, activation="relu")(g)
        g = layers.Dense(c, activation="sigmoid")(g)
        return layers.Multiply()([x, g])

    # ------------------------------------------------------
    # Squeeze-and-Excitation
    # ------------------------------------------------------
    def SE(x, r=8):
        c = x.shape[-1]
        s = layers.GlobalAveragePooling2D()(x)
        s = layers.Dense(c // r, activation="relu")(s)
        s = layers.Dense(c, activation="sigmoid")(s)
        return layers.Multiply()([x, s])

    # ------------------------------------------------------
    # Global Context Block
    # ------------------------------------------------------
    def GC_Block(x, reduction=8):
        c = x.shape[-1]
        g = layers.GlobalAveragePooling2D()(x)
        g = layers.Dense(c // reduction, activation="relu")(g)
        g = layers.Dense(c, activation="sigmoid")(g)
        return layers.Multiply()([x, g])

    # ------------------------------------------------------
    # Bloc hybride combinant toutes les attentions
    # ------------------------------------------------------
    def HybridBlock(x):
        y = ConvNeXtBlock(x)
        y = LCA(y)
        y = SE(y)
        y = GC_Block(y)
        return layers.Add()([x, y])

    # ------------------------------------------------------
    # Entrée du réseau
    # ------------------------------------------------------
    inp = keras.Input((19, 19, planes))

    # Stem convolutionnel
    x = layers.Conv2D(filters, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Empilement de 6 blocs hybrides
    for _ in range(6):
        x = HybridBlock(x)

    # ------------------------------------------------------
    # Policy head : distribution des coups
    # ------------------------------------------------------
    p = layers.Conv2D(1, 1, padding="same")(x)
    p = layers.Flatten()(p)
    p = layers.Activation("softmax", name="policy")(p)

    # ------------------------------------------------------
    # Value head : estimation de la valeur de la position
    # ------------------------------------------------------
    v = layers.GlobalAveragePooling2D()(x)
    v = layers.Dense(128, activation="relu", kernel_regularizer=l2)(v)
    v = layers.Dense(1, activation="sigmoid", name="value")(v)

    # Modèle final
    model = keras.Model(inp, [p, v])
    
    # Vérification du nombre de paramètres
    total = model.count_params()
    print("Total params:", total)
    assert total < 100_000, f"Modèle trop gros ({total})"

    return model

# Création du modèle
model = build_model()

# Affichage du résumé de l’architecture
model.summary()

# ==========================================================
# 4. Compilation
# ==========================================================
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss={
        "policy": "categorical_crossentropy",
        "value": "mse"
    },
    loss_weights={
        "policy": 1.0,
        "value": 0.4
    },
    metrics={
        "policy": "accuracy",
        "value": "mae"
    }
)

# ==========================================================
# 5. Buffers de données
# ==========================================================
input_data = np.zeros((N, 19, 19, planes), dtype="float32")
policy = np.zeros((N, moves), dtype="float32")
value = np.zeros((N,), dtype="float32")
end = np.zeros((N, 19, 19, 2), dtype="float32")
groups = np.zeros((N, 19, 19, 1), dtype="float32")

# Value remise en forme pour Keras
value_reshaped = value.reshape(-1, 1)

# ==========================================================
# 6. Historique des métriques
# ==========================================================
hist = {
    "epoch": [],
    "total_loss": [],
    "policy_loss": [],
    "value_loss": [],
    "policy_acc": [],
    "value_mae": [],
}

# ==========================================================
# 7. Pré-training policy-only
# ==========================================================
print("Pré-training policy-only")

# Chargement validation initiale
golois.getValidation(input_data, policy, value, end)

for epoch in range(1, 201):
    print(f"\nEpoch {epoch}/200 (policy-only)")

    # Chargement batch
    golois.getBatch(input_data, policy, value, end, groups, epoch * N)

    # Augmentation
    X_aug, P_aug = augment_batch_by_symmetry(input_data, policy)

    # Entraînement (value head présente mais non prioritaire)
    model.fit(
        X_aug, [P_aug, value_reshaped],
        batch_size=batch, epochs=1, verbose=1
    )

    # Évaluation périodique
    if epoch % 10 == 0:
        val = model.evaluate(
            X_aug, [P_aug, value_reshaped],
            batch_size=batch, verbose=0
        )

        print(
            f"[POLICY-ONLY VAL] "
            f"total_loss={val[0]:.4f} | "
            f"policy_loss={val[1]:.4f} | "
            f"value_loss={val[2]:.4f} | "
            f"policy_acc={val[3]:.4f} | "
            f"value_mae={val[4]:.4f}"
        )

# ==========================================================
# 8. Boucle d'entraînement complet (policy + value)
# ==========================================================
print("Training complet")

for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")

    golois.getBatch(input_data, policy, value, end, groups, epoch * N)
    X_aug, P_aug = augment_batch_by_symmetry(input_data, policy)

    model.fit(
        X_aug, [P_aug, value_reshaped],
        batch_size=batch, epochs=1, verbose=1
    )

    if epoch % 10 == 0:
        val = model.evaluate(
            X_aug, [P_aug, value_reshaped],
            batch_size=batch, verbose=0
        )

        # Sauvegarde des métriques
        hist["epoch"].append(epoch)
        hist["total_loss"].append(val[0])
        hist["policy_loss"].append(val[1])
        hist["value_loss"].append(val[2])
        hist["policy_acc"].append(val[3])
        hist["value_mae"].append(val[4])

        print(
            f"[VAL] "
            f"total_loss={val[0]:.4f} | "
            f"policy_loss={val[1]:.4f} | "
            f"value_loss={val[2]:.4f} | "
            f"policy_acc={val[3]:.4f} | "
            f"value_mae={val[4]:.4f}"
        )

    # Nettoyage mémoire
    if epoch % 5 == 0:
        gc.collect()

# ==========================================================
# 9. Sauvegarde du modèle
# ==========================================================
model.save("model.h5")
print("Modèle sauvegardé")

# ==========================================================
# 10. Génération et sauvegarde des graphiques
# ==========================================================
def save_plot(x, y, title, filename):
    """
    Génère un graphique simple et le sauvegarde sur disque.
    """
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.grid()
    plt.savefig(filename)
    plt.close()

save_plot(hist["epoch"], hist["policy_acc"],  "Policy Accuracy", "policy_accuracy.png")
save_plot(hist["epoch"], hist["policy_loss"], "Policy Loss",     "policy_loss.png")
save_plot(hist["epoch"], hist["value_loss"],  "Value Loss",      "value_loss.png")
save_plot(hist["epoch"], hist["value_mae"],   "Value MAE",       "value_mae.png")
save_plot(hist["epoch"], hist["total_loss"],  "Total Loss",      "total_loss.png")

print("Plots sauvegardés (.png)")
