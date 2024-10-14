import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def plot_pie_chart(data: pd.DataFrame, column: str, title: str) -> None:
    value_counts = data[column].value_counts()
    total = sum(value_counts)
    plt.figure(figsize=(10, 10))
    wedges, texts, auto_texts = plt.pie(
        value_counts,
        labels=value_counts.index,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(value_counts) / 100)})' if p > 2 else '',
        startangle=90,
        pctdistance=0.85,
        labeldistance=1.1,
        textprops={'fontsize': 11},
        shadow=True
    )
    for autotext in auto_texts:
        autotext.set_fontsize(12)
        autotext.set_color('white')
    legend_labels = [
        f'{label} ({value_counts[label]}, {value_counts[label] / total * 100:.1f}%)'
        for label in value_counts.index
    ]
    plt.legend(wedges, legend_labels, title=column, loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    plt.title(title)
    plt.show()


def plot_confusion_matrix(ground_truth, predictions, title) -> None:
    cm = confusion_matrix(ground_truth, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()
