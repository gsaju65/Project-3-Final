---
title: Drone Object Classifier
emoji: 🚁
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Classify objects in drone images with a model on Cloud Run
---

# Drone Object Classifier

Pick a crop from a drone image, or upload your own, and a model running on Google Cloud Run guesses whether it's a person, a car or van, a truck or bus, or a two-wheeler.

This Space is only the front-end. It sends each image to the model API's `/predict` endpoint and shows the class probabilities. The data pipeline, training, MLflow tracking, Docker setup and API code are on [GitHub](https://github.com/gsaju65/Project-3-Final).

The examples are crops from VisDrone flights the model never trained on. It gets about 77% of those right, and trucks are its weak spot.