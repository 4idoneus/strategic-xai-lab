# **Beyond The Move: An Explainable AI (XAI) Framework for Cognitive Skill Acquisition in the Game of Go**

## **Project Overview**

**Beyond The Move (BM)** is an ongoing research project exploring how Explainable AI (XAI) methods can be applied to strategic games—starting with the game of Go—to generate human-like move recommendations and interpretable explanations. The project is currently in active development.

## **Research Question**

**How can Explainable AI methods be used to make AI move recommendations in strategy games more interpretable and human-like?**

## **Research Goal**

To create a **“Glass Box” AI** that visualises its strategic reasoning, helping human players improve their intuition rather than just memorising computer moves.  
The goal is to create an XAI model that **teaches rather than competes**.

---

## **Project Timeline**

---

### **Phase I — CNN, “Professional Mimic” (Current Phase)**

Development of a system utilising **Residual Neural Networks (ResNet)**.  
The model demonstrates excellent intuition regarding the **Shape** and **Direction of Play**, but possesses limited calculation capabilities for deep fighting.

- **Status:** Active Development  
- **Key Tech:** PyTorch, ResNet-18, Integrated Gradients

---

### **Phase II — Topological Learning & GNNs**

Transitioning from pixel-based Convolutional Neural Networks to **Graph Neural Networks (GNNs)**.  
This phase aims to model the board as a **graph of connected stones** to better capture topological properties and **“Aji” (latent potential)**, moving beyond static shape recognition.

**Iterative Refinement:**  
This phase includes continuous benchmarking against Phase I baselines.  
The graph architecture will undergo iterative adjustments to maximise its ability to represent stone connectivity compared to standard CNNs.

---

### **Phase III — Generalisation & Cognitive Transfer**

This phase expands the scope to prepare for **advanced Master's-level research**.

#### **Cross-Domain Generalisation**
Extending the XAI model to **Shogi** and **Chess** to validate the universality of “Interaction Primitives” across strategic games.

#### **User Modelling (IRL)**
Implementing **Inverse Reinforcement Learning** to model individual player styles.  
This shifts the AI from a generic tutor to a personalised **“Sensei”** that understands the specific biases of the user.

#### **Counterfactual Explanations**
Developing the capability to answer **“Why not?”** questions  
(e.g., *“Why is my move worse than the AI's move?”*), which is critical for cognitive skill acquisition.

#### **Adaptive Roadmap**
The specific methodologies in this phase are subject to evolution, allowing for flexible adaptation based on the structural findings and technical breakthroughs achieved in Phase II.

---

## **Acknowledgments**

Special thanks to **@featurecat** for the Fox Go Dataset.
