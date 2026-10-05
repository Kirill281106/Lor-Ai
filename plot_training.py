import matplotlib.pyplot as plt

epochs = list(range(1, 16))
train_acc = [0.7560, 0.9174, 0.9323, 0.9480, 0.9489, 0.9600, 0.9625, 0.9702, 0.9664, 0.9710, 0.9698, 0.9710, 0.9727, 0.9791, 0.9749]
val_acc   = [0.8995, 0.9319, 0.9404, 0.9404, 0.9540, 0.9472, 0.9472, 0.9625, 0.9676, 0.9710, 0.9642, 0.9693, 0.9744, 0.9727, 0.9710]
train_loss= [0.7755, 0.3415, 0.2567, 0.2047, 0.1877, 0.1588, 0.1474, 0.1307, 0.1210, 0.1121, 0.1084, 0.1013, 0.0974, 0.0894, 0.0907]
val_loss  = [0.4000, 0.2741, 0.2181, 0.2069, 0.1686, 0.1696, 0.1565, 0.1313, 0.1202, 0.1203, 0.1111, 0.1201, 0.1006, 0.1069, 0.0977]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_acc, 'b-o', label='Train Accuracy', linewidth=2)
ax1.plot(epochs, val_acc, 'r-o', label='Validation Accuracy', linewidth=2)
ax1.set_xlabel('Эпоха', fontsize=12)
ax1.set_ylabel('Точность', fontsize=12)
ax1.set_title('Точность модели по эпохам', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2)
ax2.plot(epochs, val_loss, 'r-o', label='Validation Loss', linewidth=2)
ax2.set_xlabel('Эпоха', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Функция потерь по эпохам', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ График сохранен: training_history.png")