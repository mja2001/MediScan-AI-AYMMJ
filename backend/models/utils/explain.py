import shap
import torch
from captum.attr import IntegratedGradients, GradCAM
import matplotlib.pyplot as plt

def generate_heatmap(model, input_tensor, target_class):
    gradcam = GradCAM(model=model, target_layers=[model.base.features[-1]])
    attributions = gradcam.attribute(input_tensor, target=target_class)
    # Visualize
    plt.imshow(attributions[0].cpu().detach().numpy(), cmap='hot')
    plt.savefig('heatmap.png')
    return 'heatmap.png'

def get_shap_values(model, data):
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)
    return shap_values

def confidence_score(preds):
    # Calibration
    return preds.max().item()
