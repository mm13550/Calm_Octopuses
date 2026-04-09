This plan outlines the architecture for a cross-modal "Taste Profile" recommendation engine. Since you are already working with Python and libraries like PyTorch and NumPy, this workflow leverages those tools to move from general data to a personalized NYC Michelin experience.

---

## Phase 1: The Training Foundation (Yelp Sandbox)
Before touching the NYC data, you must build and validate the "logic" of your model using the **Yelp Open Dataset**.

1.  **Preprocessing:**
    * Filter the `review.json` for users with $\geq 10$ reviews to ensure a rich history.
    * Clean text data (tokenization, removing stop words).
2.  **Cross-Modal Embedding Engine:**
    * **Text Tower:** Use a pre-trained Transformer (like `DistilBERT`) to convert reviews into vectors.
    * **Image Tower:** Use a Vision Transformer (ViT) or ResNet to embed dish photos.
    * **Fusion:** Use an **Autoencoder** structure to compress these high-dimensional vectors into a shared "Latent Taste Space" ($\mathbf{z}$).
    * **Objective:** Minimize the distance between a dish photo and its corresponding review text in the latent space.



---

## Phase 2: The NYC Michelin Layer
Now, you apply that pre-trained logic to your specific niche.

1.  **Niche Data Acquisition:**
    * Use the **Google Places API** to pull the 10 reviews and photos for the ~500 NYC Michelin restaurants.
    * **Entity Matching:** Use `fuzzywuzzy` to join the Michelin Guide names with Google Place IDs.
2.  **Profiling:**
    * For each restaurant, pass its 10 reviews and photos through your pre-trained **Embedding Engine**.
    * **The Centroid:** Calculate the mean vector of these 10 reviews to create the **Restaurant Taste Profile** ($\mathbf{r}$).
    * Store these profiles in a vector database (like `FAISS` or `ChromaDB`) for fast retrieval.

---

## Phase 3: Personalization & Regression
This is where you build the "Predictor" for your app users.

1.  **User Vector Construction:**
    * When a user rates a restaurant in your app, fetch that restaurant's pre-calculated profile ($\mathbf{r}$).
    * Calculate the **User Vector** ($\mathbf{u}$) as a weighted average of their history:
        $$\mathbf{u} = \frac{\sum_{i=1}^{n} w_i \mathbf{r}_i}{\sum w_i}$$
        *(Where $w_i$ can be a decay factor to prioritize recent visits).*
2.  **The Regression Head:**
    * Construct a Feed-Forward Neural Network in PyTorch.
    * **Input:** Concatenated vector $[\mathbf{u} ; \mathbf{r}_{target} ; \text{metadata}]$.
    * **Output:** A single scalar representing the predicted star rating $\hat{y} \in [1, 5]$.
    * **Loss Function:** Use **Mean Squared Error (MSE)** to optimize the weights.



---

## Phase 4: Validation (The "Time-Split" Test)
To ensure this works before "launching," go back to your Yelp data for a final sanity check.

1.  **Simulate User History:** For a "Heavy User," hide their 3 most recent reviews.
2.  **Predict:** Use the model to predict the rating for those 3 "unseen" restaurants.
3.  **Evaluate:**
    * **Accuracy:** Is the Mean Absolute Error (MAE) $< 0.8$?
    * **Hit Rate:** If the user gave a 5-star rating, did your model predict at least a 4.2?

---

## Phase 5: Implementation (The Pythonic Way)
To get this up and running in your environment:

* **Data Handling:** Use `pandas` for the initial Yelp JSON parsing and `NumPy` for the vector operations.
* **Modeling:** Use `PyTorch Lightning` to organize the training loops for the autoencoder and regression head.
* **Inference:** Once the weights are trained, export the model as a `.pt` file for use in your app backend.

---
