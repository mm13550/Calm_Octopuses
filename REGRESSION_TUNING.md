git ignore this file

Lets focus on the features that we want again.

I want the model to be more restaurant focused than user focused.
Therefore we should zero out user features sometimes, lets start with 25% of the time.
The interaction features is also important. We will not zero out interaction features.
However, metadata can lead the model to become unopinionated. Lets also zero out metadata 25% of the time.
The only data that should not experience any droppout is the restaurant features.

The data should also be normalized and standarized.
We should use centroid subtraction because of data being highly compact in high dimensions.

We want the model to be opinionated, so we will use a low temperature whenever appropriate.

