{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMmB2d3w67yHR/3edNMIXYW",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/isratrimii/Fake-Job-Detection/blob/main/Fake_Job_Detection.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RKoiERkHwKko"
      },
      "outputs": [],
      "source": [
        "!pip install gradio\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.utils.class_weight import compute_class_weight\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics import confusion_matrix, classification_report\n",
        "\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Dropout\n",
        "\n",
        "from google.colab import files"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Upload dataset"
      ],
      "metadata": {
        "id": "mD28AgxcwPZJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "uploaded = files.upload()\n",
        "\n",
        "df = pd.read_csv(\"fake_job_postings.csv\")\n",
        "\n",
        "df = df[[\"description\", \"fraudulent\"]]\n",
        "df[\"description\"] = df[\"description\"].fillna(\"\")"
      ],
      "metadata": {
        "id": "6qguOPlNwVXh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "NLP Feature Engineering"
      ],
      "metadata": {
        "id": "fAzTiYjSwa2_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_text = df[\"description\"].astype(str).values\n",
        "y = df[\"fraudulent\"].values\n",
        "\n",
        "tfidf = TfidfVectorizer(max_features=6000, stop_words=\"english\")\n",
        "X = tfidf.fit_transform(X_text).toarray()"
      ],
      "metadata": {
        "id": "fimDlPMWwbh4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Train-Test Split"
      ],
      "metadata": {
        "id": "fNGj1yfRwgsC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42\n",
        ")\n",
        "\n",
        "class_weights = compute_class_weight(\n",
        "    class_weight=\"balanced\",\n",
        "    classes=np.unique(y_train),\n",
        "    y=y_train\n",
        ")\n",
        "\n",
        "class_weights = dict(enumerate(class_weights))"
      ],
      "metadata": {
        "id": "GWdk7LNYwhNe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = Sequential([\n",
        "    Dense(512, activation=\"relu\", input_shape=(6000,)),\n",
        "    Dropout(0.5),\n",
        "\n",
        "    Dense(256, activation=\"relu\"),\n",
        "    Dropout(0.4),\n",
        "\n",
        "    Dense(128, activation=\"relu\"),\n",
        "    Dropout(0.3),\n",
        "\n",
        "    Dense(1, activation=\"sigmoid\")\n",
        "])\n",
        "\n",
        "model.compile(\n",
        "    optimizer=\"adam\",\n",
        "    loss=\"binary_crossentropy\",\n",
        "    metrics=[\"accuracy\"]\n",
        ")"
      ],
      "metadata": {
        "id": "s-ipnXlIwnPO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Train Model"
      ],
      "metadata": {
        "id": "PTK7FMJuwq25"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(\n",
        "    X_train, y_train,\n",
        "    epochs=12,\n",
        "    batch_size=32,\n",
        "    validation_split=0.2,\n",
        "    class_weight=class_weights\n",
        ")"
      ],
      "metadata": {
        "id": "-twd_ZJ-wrYK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Evaluation"
      ],
      "metadata": {
        "id": "4T33JgfZww-E"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "loss, acc = model.evaluate(X_test, y_test)\n",
        "print(\"🔥 Accuracy:\", acc)\n",
        "\n",
        "y_pred = (model.predict(X_test) > 0.5).astype(int)\n",
        "\n",
        "print(\"\\n📊 Confusion Matrix:\")\n",
        "print(confusion_matrix(y_test, y_pred))\n",
        "\n",
        "print(\"\\n📄 Classification Report:\")\n",
        "print(classification_report(y_test, y_pred))"
      ],
      "metadata": {
        "id": "gS8heRCtwxjh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Smart Prediction Function"
      ],
      "metadata": {
        "id": "nx_XzvOCw3Ha"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def predict_job(text):\n",
        "    vec = tfidf.transform([text]).toarray()\n",
        "    pred = model.predict(vec)[0][0]\n",
        "\n",
        "    print(\"\\n🧠 AI Score:\", pred)\n",
        "\n",
        "    if pred > 0.7:\n",
        "        print(\"❌ Very Likely FAKE JOB\")\n",
        "    elif pred > 0.4:\n",
        "        print(\"⚠️ Suspicious Job\")\n",
        "    else:\n",
        "        print(\"✅ Likely REAL JOB\")"
      ],
      "metadata": {
        "id": "Y2LJd50Gw3th"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Test Example"
      ],
      "metadata": {
        "id": "W7BW43h9xAbk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "predict_job(\"Work from home job. No experience needed. Earn 5000 dollars easily.\")"
      ],
      "metadata": {
        "id": "3XK0wgTsxA9R"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Web App"
      ],
      "metadata": {
        "id": "bSQmZWY-xHzX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import gradio as gr\n",
        "\n",
        "def predict_ui(text):\n",
        "    vec = tfidf.transform([text]).toarray()\n",
        "    pred = model.predict(vec)[0][0]\n",
        "\n",
        "    if pred > 0.7:\n",
        "        return \"❌ FAKE JOB\"\n",
        "    elif pred > 0.4:\n",
        "        return \"⚠️ SUSPICIOUS JOB\"\n",
        "    else:\n",
        "        return \"✅ REAL JOB\"\n",
        "\n",
        "app = gr.Interface(\n",
        "    fn=predict_ui,\n",
        "    inputs=\"text\",\n",
        "    outputs=\"text\",\n",
        "    title=\"Fake Job Detection AI\",\n",
        "    description=\"Paste job description and check if it's real or fake\"\n",
        ")\n",
        "\n",
        "app.launch()"
      ],
      "metadata": {
        "id": "ld8y_ha6xKzC"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}