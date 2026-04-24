# Retrieval Hardening Handoff

## 1. Purpose of This Branch

This handoff explains the retrieval changes made on branch:

- `retrieval-hardening`

The goal of this branch was **not** to redesign Leo's retrieval architecture.
The goal was to make the existing retrieval layer more reliable for downstream use, especially for:

- Merry's frontend integration
- stable joins against curated restaurant metadata
- local execution in environments where full ML dependencies may be missing

This branch is intentionally narrow in scope.

Compared to `main`, it changes only:

- `algorithms/retrieval_engine.py`
- `algorithms/retrieval_engine_reviews.py`
- `algorithms/retrieval_engine_profiles.py`
- `algorithms/retrieval_engine_hybrid.py`
- `algorithms/retrieval_metadata.py`
- this handoff file

---

## 2. What Stayed The Same

This branch keeps Leo's original high-level retrieval structure:

- menu retrieval remains in `algorithms/retrieval_engine.py`
- review retrieval remains in `algorithms/retrieval_engine_reviews.py`
- hybrid retrieval remains in `algorithms/retrieval_engine_hybrid.py`
- restaurant-level profile retrieval remains in `algorithms/retrieval_engine_profiles.py`

This branch does **not**:

- change the LanceDB table layout
- change upstream embedding generation pipelines
- change `restaurant_profiles_latest.jsonl`
- redesign the recommendation methodology

So this should be understood as a **hardening / interface cleanup pass**, not a backend rewrite.

---

## 3. Main Problems In The Original Retrieval Layer

Based on Leo's original retrieval scripts, the main downstream issues were:

1. Results were not fully frontend-ready.
Many outputs returned `restaurant_name`, but did not consistently return:
- `restaurant_id`
- `homepage`
- `borough`
- `michelin_category`

2. The retrieval layer depended too heavily on local ML environment setup.
Some modules imported `torch` / `transformers` at import time, which made them fragile on machines that did not have those packages installed.

3. Some logic still reflected older project assumptions.
The project has already removed `time-decay` from scope, but older decay-related structures and ranking ideas were still present in retrieval code.

4. Hybrid merging was not stable enough for UI work.
Merging by `restaurant_name` is weaker than merging by `restaurant_id`, especially for frontend use.

5. Profile retrieval was the best frontend candidate, but the output contract was still too thin.
Merry needs richer restaurant-card fields than just a name, score, and snippets.

---

## 4. Concrete Changes Made

### 4.1 Added a shared curated metadata join layer

New file:

- `algorithms/retrieval_metadata.py`

This file introduces:

- `load_restaurant_lookup(...)`
- `get_restaurant_metadata(...)`

It reads:

- `data/csv/restaurant_lookup.csv`

and provides a shared way for retrieval scripts to attach:

- `restaurant_id`
- `restaurant_name`
- `homepage`
- `borough`
- `michelin_category`

### Why this was added

Leo's retrieval outputs were good for backend experimentation, but Merry needs stable, curated metadata for cards and links.
This change ensures retrieval results are tied back to the manually corrected lookup table rather than forcing downstream teammates to perform their own joins.

---

### 4.2 Hardened menu retrieval

Updated file:

- `algorithms/retrieval_engine.py`

Main changes:

- removed real ranking dependence on time-decay
- removed the old default-query-vector behavior
- added lazy CLIP backend loading
- added lexical fallback when CLIP dependencies are unavailable
- added automatic table initialization from `data/embeddings/menu_embeddings_latest.jsonl`
- added curated metadata fields to returned results
- changed restaurant deduplication to prefer `restaurant_id`

The returned simple payload now includes fields such as:

- `doc_id`
- `restaurant_id`
- `restaurant_name`
- `homepage`
- `borough`
- `michelin_category`
- `dish_name`
- `price`
- `score`
- `semantic_similarity`
- `lexical_bonus`
- `content_type`
- `source`
- `text`

### Why these changes were made

The previous version still behaved like a prototype in two important ways:

1. It was not always using the user's query in a robust retrieval path.
2. It did not return enough structured metadata for direct frontend use.

This update makes menu retrieval usable as a real downstream interface instead of just a local backend experiment.

---

### 4.3 Hardened review retrieval

Updated file:

- `algorithms/retrieval_engine_reviews.py`

Main changes:

- removed real ranking dependence on time-decay / freshness logic
- moved CLIP dependency loading behind runtime checks
- added lexical fallback review retrieval
- added automatic table initialization from `data/embeddings/review_embeddings_latest.jsonl`
- attached curated metadata to results
- exposed `backend_used` in the API payload

Returned result items now include:

- `doc_id`
- `restaurant_id`
- `restaurant_name`
- `homepage`
- `borough`
- `michelin_category`
- `rating`
- `score`
- `semantic_similarity`
- `lexical_bonus`
- `source`
- `text`

### Why these changes were made

Review retrieval is especially useful as a supporting evidence layer for frontend cards and detail views.
That makes runtime stability more important than preserving older experimental ranking logic.

This update ensures that:

- review retrieval can still run without a full CLIP stack
- results are joinable and displayable downstream

---

### 4.4 Upgraded restaurant profile retrieval into a frontend-facing contract

Updated file:

- `algorithms/retrieval_engine_profiles.py`

This was the most important frontend-facing change.

Main changes:

- added lazy CLIP loading
- added lexical fallback profile retrieval
- added automatic table initialization from `data/embeddings/restaurant_profiles_latest.jsonl`
- attached curated metadata
- expanded result payload to include image paths and restaurant-card content

Returned profile results now include:

- `doc_id`
- `restaurant_id`
- `restaurant_name`
- `homepage`
- `borough`
- `michelin_category`
- `score`
- `semantic_similarity`
- `lexical_bonus`
- `menu_item_count`
- `review_count`
- `food_image_count`
- `interior_image_count`
- `food_image_paths`
- `interior_image_paths`
- `top_menu_items`
- `top_review_snippets`
- `content_type`
- `source`
- `text`

### Why these changes were made

Leo's profile retrieval was already the strongest candidate for frontend search.
However, Merry still would not have had enough structured fields to render restaurant cards cleanly.

This branch keeps the original profile-retrieval direction but turns the payload into something closer to a real frontend contract.

---

### 4.5 Made hybrid retrieval safer for frontend merging

Updated file:

- `algorithms/retrieval_engine_hybrid.py`

Main changes:

- normalizes menu and review payloads into richer structured rows
- merges restaurants by `restaurant_id` first, then falls back to name
- preserves curated metadata at the restaurant-card level
- keeps both `menu_matches` and `review_matches`
- adds `has_menu_signal` and `has_review_signal`
- exposes `menu_backend_used` and `review_backend_used`

### Why these changes were made

Hybrid retrieval is most useful when a frontend wants to show:

- one restaurant card
- menu evidence
- review evidence

Merging by `restaurant_name` alone is not stable enough for that purpose.
This change reduces the chance of duplicate or mis-merged restaurant cards.

---

## 5. Scope Clarification: Time-Decay

This branch removes real ranking dependence on time-decay.

Important note:

- some neutral compatibility fields such as `decay_factor`, `freshness_adjustment`, and `trending_badge` may still exist in certain dataclasses
- those fields are no longer driving ranking behavior
- they are now effectively placeholders only

This was done because current project scope documents already state that:

- `time-decay` is out of scope
- retrieval should be based on relevance, not timestamp weighting

---

## 6. Why The Keyword Lists Were Kept

The retrieval scripts still contain explicit keyword lists such as:

- `TOKEN_ALIASES`
- `STRICT_TERMS`
- `STYLE_TERMS`
- `POSITIVE_STYLE_TERMS`
- `NEGATION_TERMS`

These were intentionally kept because they still serve an important purpose:

1. They improve deterministic query normalization.
Example:
- `prawn` can map to `shrimp`
- `tagliatelle` can map to `pasta`
- `cozy` can map to `quiet`

2. They make lexical fallback much more usable.
If CLIP is unavailable locally, retrieval still needs basic guardrails.

3. They support light filtering and reranking behavior.
Examples:
- `vegetarian` can exclude obviously meat-heavy rows
- exact dish or vibe terms can receive lexical bonus

So these lists are not replacing embedding retrieval.
They are acting as a controlled backup layer and a lightweight reranking mechanism.

---

## 7. What Was Tested

The updated retrieval scripts were tested locally in an environment where CLIP dependencies were not fully available.

The following kinds of queries were checked:

- `uni pasta`
- `friendly staff`
- `quiet omakase`
- `wine bar`
- `seafood tasting`

The goal of this testing was to verify:

- modules can import without failing
- fallback retrieval actually returns usable results
- result payloads now include curated metadata needed downstream

---

## 8. Recommended Usage After This Branch

### For Merry

Recommended starting points:

- use `search_profiles_api(...)` for main restaurant-card search
- use `hybrid_search_api(...)` for evidence expansion or detail views

Why:

- profile retrieval now returns enough restaurant-card fields to power a frontend search page
- hybrid retrieval now gives a cleaner restaurant-level merge of menu and review signals

### For Leo

This branch should be read as:

- preserving Leo's architecture
- tightening the retrieval contract
- aligning runtime behavior with current project scope

It is not intended as a criticism of the original backend work.
It is a downstream integration pass.

---

## 9. Summary

In one sentence:

This branch keeps Leo's original retrieval layout, but makes it safer, more joinable, less dependent on local ML setup, and more useful for Merry's frontend integration.
