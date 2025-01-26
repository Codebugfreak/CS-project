import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import imageio
from project import normalize_path

# Layer metadata for each model
LAYER_METADATA = {
    "InceptionV3": {
        "light": "mixed3",
        "medium": "mixed7",
        "heavy": "mixed10"
    },
    "VGG16": {
        "subtle": "block1_conv2",
        "balanced": "block3_conv3",
        "intense": "block5_conv3"
    },
    "ResNet50": {
        "sparse": "conv1_relu",
        "focused": "conv4_block6_out",
        "dense": "conv5_block3_out"
    }
}

def preprocess_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error reading image file: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0).astype('float32') / 255.0
    return tf.convert_to_tensor(img)

def load_model(model_name, layer_name):
    try:
        if model_name == "InceptionV3":
            model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
        elif model_name == "VGG16":
            model = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        elif model_name == "ResNet50":
            model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Extract the specified layer
        layer = model.get_layer(layer_name)
        return tf.keras.Model(inputs=model.input, outputs=layer.output)
    except Exception as e:
        raise ValueError(f"Error loading model or layer: {e}")

def multi_layer_dreaming(model, img, gradient_multiplier=0.01, iterations=1):
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(img)
            activations = model(img)
            loss = tf.reduce_mean(activations)
        grads = tape.gradient(loss, img)
        img.assign_add(grads * gradient_multiplier)
    return img

def generate_animation(image_path, model_name, layer_name, output_path="static/generated/animation.gif", frames=10):
    try:
        img = tf.Variable(preprocess_image(image_path))
        model = load_model(model_name, layer_name)

        frame_list = []
        for _ in range(frames):
            img = multi_layer_dreaming(model, img, gradient_multiplier=0.01, iterations=1)
            img_np = (tf.clip_by_value(img, 0.0, 1.0).numpy()[0] * 255).astype("uint8")
            frame_list.append(img_np)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frame_list, duration=0.2)
        return normalize_path(output_path).split("static/")[-1]
    except Exception as e:
        print(f"Error generating animation: {e}")
        return None

def generate_visual(image_path, model_name="InceptionV3", layer_name="mixed10", gradient_multiplier=0.01, iterations=1, apply_filter=None, animation=False):
    try:
        uploads_dir = Path('static/uploads').resolve()
        image_path_resolved = Path(image_path).resolve()
        if not uploads_dir in image_path_resolved.parents:
            raise PermissionError("Invalid file path.")

        if animation:
            return generate_animation(image_path, model_name, layer_name)
        else:
            img = tf.Variable(preprocess_image(image_path))
            model = load_model(model_name, layer_name)
            img = multi_layer_dreaming(model, img, gradient_multiplier, iterations)
            img = tf.clip_by_value(img, 0.0, 1.0)
            img = (img.numpy()[0] * 255).astype("uint8")

            if apply_filter == "sepia":
                img = apply_sepia(img)
            elif apply_filter == "grayscale":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            output_dir = "static/generated"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, img)
            return output_path
    except Exception as e:
        print(f"Error in generate_visual: {e}")
        return None

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(image, kernel)
