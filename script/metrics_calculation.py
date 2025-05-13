def evaluation(model,X_test, y_test,model_name):
    from sklearn.metrics import f1_score, roc_auc_score, classification_report, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    # Evaluation
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    #drawing the roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curve', fontsize=14, weight='bold')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/Users/anhnguyendo/Documents/Python machine learning/Credit risk and loan default prediction/credit-risk-analysis-and-loan-default-prediction/figures/ROC_curve_{model_name}.png', dpi=300)
    plt.show()

    #drawing the precision curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, color='darkorange', label=f'Precision-Recall (AP = {avg_precision:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{model_name} - Precision-Recall Curve', fontsize=14, weight='bold')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/Users/anhnguyendo/Documents/Python machine learning/Credit risk and loan default prediction/credit-risk-analysis-and-loan-default-prediction/figures/precision_curve_{model_name}.png', dpi=300)
    plt.show()

    #drawing the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',  # integer format for count values
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        annot_kws={"size": 14}
    )

    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, weight='bold')
    ax.xaxis.set_ticklabels(['Did Not Default', 'Default'], fontsize=11)
    ax.yaxis.set_ticklabels(['Did Not Default', 'Default'], fontsize=11)
    
    plt.tight_layout()  # Avoid label cutoffs
    plt.savefig(f'/Users/anhnguyendo/Documents/Python machine learning/Credit risk and loan default prediction/credit-risk-analysis-and-loan-default-prediction/figures/confusion_matrix_{model_name}.png', dpi=300)
    plt.show()

    return {
    f'f1_score_{model_name}': f1_score(y_test, y_pred),
    f'roc_auc_{model_name}': auc_score,
    f'average_precision_{model_name}': avg_precision
}
