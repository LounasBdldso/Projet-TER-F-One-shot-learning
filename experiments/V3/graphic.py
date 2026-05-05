import matplotlib.pyplot as plt
import numpy as np

# Données extraites des logs du second entraînement
epochs = list(range(1, 150))

train_loss = [
    3.3095, 2.1360, 1.7685, 1.5862, 1.4726, 1.3907, 1.3351, 1.2614, 1.2264, 1.1800,
    1.1500, 1.1322, 1.0967, 1.0651, 1.0522, 1.0171, 1.0095, 0.9859, 0.9864, 0.9576,
    0.9427, 0.9261, 0.9106, 0.8993, 0.8726, 0.8526, 0.8641, 0.8627, 0.8504, 0.8521,
    0.8083, 0.8083, 0.8060, 0.7836, 0.7922, 0.7996, 0.7859, 0.7637, 0.7575, 0.7593,
    0.7423, 0.7361, 0.7533, 0.7167, 0.7094, 0.7129, 0.7010, 0.6981, 0.6880, 0.6889,
    0.6796, 0.6860, 0.6734, 0.6686, 0.6442, 0.6729, 0.6496, 0.6573, 0.6291, 0.6298,
    0.6443, 0.6316, 0.6333, 0.6113, 0.6085, 0.6018, 0.6017, 0.6103, 0.6161, 0.5818,
    0.5916, 0.5812, 0.5733, 0.5612, 0.5622, 0.5712, 0.5402, 0.5631, 0.5711, 0.5492,
    0.5460, 0.5534, 0.5455, 0.5535, 0.5387, 0.5381, 0.5232, 0.5315, 0.5441, 0.5196,
    0.5335, 0.5132, 0.4518, 0.4685, 0.4423, 0.4172, 0.4096, 0.3944, 0.3886, 0.3951,
    0.3780, 0.3811, 0.3733, 0.3618, 0.3684, 0.3545, 0.3796, 0.3604, 0.3538, 0.3501,
    0.3437, 0.3452, 0.3348, 0.3310, 0.3183, 0.3093, 0.2995, 0.2983, 0.2769, 0.2830,
    0.2797, 0.2717, 0.2640, 0.2581, 0.2596, 0.2641, 0.2560, 0.2499, 0.2461, 0.2515,
    0.2490, 0.2430, 0.2457, 0.2363, 0.2379, 0.2328, 0.2131, 0.2239, 0.2201, 0.2017,
    0.2111, 0.2074, 0.2079, 0.2029, 0.2119, 0.2011, 0.1906, 0.1948, 0.1901
]

val_loss = [
    2.5326, 1.8639, 1.6483, 1.5533, 1.4611, 1.4093, 1.3587, 1.3058, 1.2372, 1.2135,
    1.1823, 1.1735, 1.1624, 1.1617, 1.1198, 1.0846, 1.1011, 1.0820, 1.0641, 1.0422,
    1.0327, 1.0261, 1.0237, 1.0505, 1.0090, 1.0027, 1.0203, 0.9756, 0.9708, 0.9663,
    0.9595, 0.9726, 0.9516, 0.9593, 0.9633, 0.9189, 0.9430, 0.9342, 0.9583, 0.9241,
    0.8995, 0.9287, 0.9100, 0.9224, 0.8937, 0.9152, 0.8989, 0.9024, 0.9389, 0.9022,
    0.9097, 0.9078, 0.8913, 0.8920, 0.8898, 0.8754, 0.8719, 0.8675, 0.8721, 0.9046,
    0.8786, 0.8916, 0.8834, 0.8985, 0.8805, 0.8806, 0.8682, 0.8711, 0.8524, 0.8627,
    0.8773, 0.8674, 0.8681, 0.8660, 0.8772, 0.8627, 0.8832, 0.8520, 0.8678, 0.8733,
    0.8360, 0.8786, 0.8780, 0.8740, 0.8774, 0.8769, 0.8702, 0.8797, 0.8577, 0.8630,
    0.8958, 0.8711, 0.7978, 0.8044, 0.8125, 0.8003, 0.7984, 0.8105, 0.7928, 0.8036,
    0.7998, 0.7997, 0.7896, 0.8197, 0.8089, 0.8060, 0.8054, 0.8365, 0.8004, 0.8259,
    0.8162, 0.8057, 0.8335, 0.8164, 0.7965, 0.7769, 0.8135, 0.8092, 0.7631, 0.7773,
    0.7900, 0.8061, 0.8166, 0.7616, 0.7973, 0.7936, 0.7966, 0.7918, 0.8080, 0.8230,
    0.8028, 0.8234, 0.8082, 0.8136, 0.7903, 0.7888, 0.8026, 0.7980, 0.8091, 0.7819,
    0.8152, 0.7961, 0.8239, 0.7993, 0.8001, 0.7877, 0.7926, 0.7942, 0.7868
]

val_acc = [
    0.3011, 0.4309, 0.4938, 0.5178, 0.5463, 0.5649, 0.5764, 0.5901, 0.6194, 0.6281,
    0.6356, 0.6416, 0.6433, 0.6439, 0.6600, 0.6729, 0.6682, 0.6715, 0.6752, 0.6838,
    0.6854, 0.6905, 0.6907, 0.6835, 0.6957, 0.6984, 0.6954, 0.7091, 0.7108, 0.7112,
    0.7101, 0.7093, 0.7173, 0.7125, 0.7111, 0.7260, 0.7184, 0.7203, 0.7181, 0.7221,
    0.7325, 0.7217, 0.7313, 0.7240, 0.7350, 0.7307, 0.7343, 0.7323, 0.7229, 0.7339,
    0.7336, 0.7307, 0.7382, 0.7364, 0.7391, 0.7439, 0.7446, 0.7420, 0.7424, 0.7324,
    0.7419, 0.7388, 0.7406, 0.7398, 0.7388, 0.7422, 0.7443, 0.7456, 0.7530, 0.7489,
    0.7462, 0.7445, 0.7535, 0.7462, 0.7438, 0.7510, 0.7412, 0.7532, 0.7469, 0.7443,
    0.7569, 0.7464, 0.7480, 0.7451, 0.7470, 0.7479, 0.7471, 0.7459, 0.7519, 0.7514,
    0.7419, 0.7495, 0.7694, 0.7691, 0.7678, 0.7716, 0.7711, 0.7674, 0.7711, 0.7716,
    0.7728, 0.7739, 0.7741, 0.7709, 0.7712, 0.7701, 0.7689, 0.7661, 0.7717, 0.7672,
    0.7689, 0.7727, 0.7671, 0.7684, 0.7751, 0.7819, 0.7719, 0.7717, 0.7833, 0.7800,
    0.7792, 0.7766, 0.7747, 0.7850, 0.7778, 0.7766, 0.7773, 0.7773, 0.7776, 0.7709,
    0.7740, 0.7750, 0.7784, 0.7747, 0.7793, 0.7796, 0.7774, 0.7787, 0.7769, 0.7847,
    0.7753, 0.7814, 0.7692, 0.7798, 0.7799, 0.7813, 0.7809, 0.7793, 0.7823
]

# Création du graphique avec deux axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Courbes des losses
color1 = 'tab:red'
color2 = 'tab:blue'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12, color='black')
ax1.plot(epochs, train_loss, color=color1, linewidth=1.5, label='Train Loss', alpha=0.8)
ax1.plot(epochs, val_loss, color=color2, linewidth=1.5, label='Val Loss', alpha=0.8)
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)

# Ajout des zones de changement de learning rate
ax1.axvspan(1, 92, alpha=0.1, color='green', label='LR = 1e-3')
ax1.axvspan(92, 114, alpha=0.1, color='yellow', label='LR = 5e-4')
ax1.axvspan(114, 135, alpha=0.1, color='orange', label='LR = 2.5e-4')
ax1.axvspan(135, 146, alpha=0.1, color='purple', label='LR = 1.25e-4')
ax1.axvspan(146, 150, alpha=0.1, color='pink', label='LR = 6.25e-5')

# Deuxième axe pour l'accuracy
ax2 = ax1.twinx()
color3 = 'tab:green'
ax2.set_ylabel('Validation Accuracy', fontsize=12, color=color3)
ax2.plot(epochs, val_acc, color=color3, linewidth=1.5, label='Val Accuracy', marker='o', markersize=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color3)
ax2.set_ylim([0.25, 0.85])

# Meilleur modèle
best_epoch = 124
best_acc = 0.7850
ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best model (epoch {best_epoch})')
ax2.plot(best_epoch, best_acc, 'r*', markersize=15, label=f'Best Val Acc: {best_acc*100:.1f}%')

# Titre et légendes
plt.title(f'Training Evolution - Second Model\nBest Validation Accuracy: {best_acc*100:.2f}% at epoch {best_epoch}', 
          fontsize=14, fontweight='bold')

# Légende combinée
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

plt.tight_layout()
plt.show()

# Affichage des métriques finales
print(f"\n{'='*50}")
print(f"RÉSUMÉ DE L'ENTRAÎNEMENT")
print(f"{'='*50}")
print(f"Configuration: Inconnue (probablement ResNet ou autre backbone)")
print(f"Early stopping: epoch 149")
print(f"Meilleur modèle: epoch {best_epoch}")
print(f"Meilleure validation accuracy: {best_acc*100:.2f}%")
print(f"{'='*50}")

# Points clés
print("\n📊 POINTS CLÉS:")
print(f"• Départ: Train Loss = 3.31, Val Acc = 30.1%")
print(f"• Après réduction LR (epoch 92): nette amélioration")
print(f"• Meilleure performance: Val Acc = {best_acc*100:.2f}% à l'époque {best_epoch}")
print(f"• Accuracy finale: {val_acc[-1]*100:.2f}%")
print(f"• Train loss final: {train_loss[-1]:.4f}")
print(f"• Val loss final: {val_loss[-1]:.4f}")