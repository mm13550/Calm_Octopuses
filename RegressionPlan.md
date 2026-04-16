Reminder that the regression model is user based. We want to track users across different restaurants, and predict their rating of a restaurant based on their past ratings and the restaurant's features.
We want this to be a quantile regression model, predicting the a 95% confidence interval of the rating distribution for a user at a restaurant.
We will use the restaurant embeddings found at data/yelp_sandbox/toy_embeddings/toy_restaurant_embeddings.pt
We'll use pinball loss to train the model.

