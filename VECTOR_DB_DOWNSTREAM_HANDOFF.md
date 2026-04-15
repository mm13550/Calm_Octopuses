# Vector Database Handoff For Downstream Teammates

## Executive Summary

Leo's overall vector database direction is sound for the current state of this repo:

- keep vectors separated by modality
- keep retrieval at the item level
- keep restaurant-level aggregation as a downstream layer

This is safer than forcing all embeddings into one shared ANN table, because the current project does not yet have one finalized shared embedding space across images, menus, reviews, and bios.

There are, however, a few important clarifications for the team:

1. The current repo does not yet contain a finalized production vector DB. The checked-in `algorithms/retrieval_engine.py` is a LanceDB prototype around a single table called `restaurant_vectors`, not the final team contract.
2. Time-decay has been removed from the current implementation plan and should be treated as deprecated older design material.
3. `restaurant_profiles.fused_vector` should be treated as optional phase-2 output, not a required v1 field. We should only create one fused vector after we agree on a valid fusion strategy.

If we follow those constraints, the design is in good shape and should be easy for downstream teammates to use.

## If You Are Not Familiar With Databases

You do not need to think about the vector database as a complicated system.

For this project, it is enough to think of it like this:

- a **table** is just a structured spreadsheet
- each **row** is one item we may want to search or join later
- a **vector** is the numeric embedding for that row
- a **search** means "find the rows whose vectors are most similar to my query"
- `rest_id` is the restaurant's unique key and is the main way all tables connect to each other

Most downstream teammates should **not** need to manage LanceDB directly.

In practice, most people will only do one of these things:

- read search results returned by Leo's helper function
- fetch all rows for one restaurant using `rest_id`
- read the one-row-per-restaurant summary table `restaurant_profiles`
- write a small number of derived fields back by `rest_id`

If it helps, you can think of the DB as having two layers:

- **item-level tables** for search
  - menu items
  - images
  - reviews
- **restaurant-level table** for downstream modeling and UI summaries
  - one row per restaurant

## Current Repo State (As Of April 15, 2026)

Important context:

- the uploaded project PDF reflects an older architecture version
- some ideas in that PDF were revised after the data mining phase
- this handoff should be treated as the current downstream-facing contract

### Canonical metadata

Use these files as the source of truth for restaurant identity and joins:

- `data/csv/seeds_resolved.csv`: curated restaurant metadata
- `data/csv/restaurant_lookup.csv`: join helper with stable `rest_id`

### Core upstream assets already available

- `data/social_reviews.csv`
  - 1,742 rows
  - 349 restaurants
  - schema: `uid, rest_id, source, text, rating`
  - current source coverage: Google Places only

- `data/social_images.csv`
  - 13,778 rows
  - 349 restaurants
  - schema: `image_uid, rest_id, source, image_path`
  - current source coverage: Yelp + Google Places

- `data/extracted_menus/final_parsed_menus.json`
  - 9,261 menu items
  - 310 restaurants
  - fields: `rest_id, restaurant_name, dish_name, ingredients, price`

- `data/extracted_bios/restaurant_bios_joinable.json`
  - 349 restaurants
  - joinable by `rest_id`

### Coverage notes

- 302 restaurants currently have reviews, images, menus, and bios all at once.
- 47 restaurants currently have reviews/images/bios but no parsed menu rows.
- 43 restaurants currently have bios but no menu rows.

This means downstream code must support partial-modality coverage. Do not assume every restaurant has every modality.

## Design Verdict

### What is good in the current design

- Separate tables by modality is the right default.
- Single-item granularity is correct for retrieval.
- A restaurant-level aggregation table is useful for Craig, Grace, and Merry.
- `rest_id` should remain the primary join key everywhere.

### What needs to be tightened

- Do not rely on one shared vector space yet.
- Do not require `created_at` on every vector row in v1.
- Do not make `fused_vector` mandatory before the fusion logic is defined.
- Do not treat the current `search_by_text()` prototype as production semantic search yet.
- Do not build any downstream dependency on time-decay.

## Why The Current Prototype Should Not Be The Team Contract

The checked-in retrieval prototype currently assumes:

- one LanceDB table: `restaurant_vectors`
- one required `vector` column for all rows
- one required `created_at` column for all rows
- time-decay during ranking

That prototype is useful for local experimentation, but it does not yet match the current dataset or the safest team-facing architecture:

- `data/social_reviews.csv` does not contain timestamps
- `data/social_images.csv` does not contain timestamps
- menu rows and bio rows do not currently come with a natural shared recency signal
- the query path still uses a placeholder vector instead of a real embedded query
- time-decay is no longer part of the current implementation plan

So the team should treat the current prototype as a sandbox, not the final downstream API.

## Recommended Vector DB Contract (v1)

### Shared rules across all vector tables

- Join on `rest_id`, never on restaurant name alone.
- Keep one embedding model per table unless a shared space has been explicitly validated.
- Store vector provenance on every table:
  - `embedding_model`
  - `embedding_dim`
  - `embedding_version`
  - `ingested_at`
- Treat `created_at` as optional metadata for now.
- Keep raw text and raw image paths available for debugging and frontend rendering.
- Do not expose time-decay fields in the stable downstream contract.

### Table: `image_vectors`

Purpose:
food-image retrieval and restaurant-level image aggregation

Recommended row grain:
one row per cleaned image

Recommended columns:

- `image_uid`
- `rest_id`
- `restaurant_name`
- `source`
- `image_path`
- `image_category`
- `keep_for_food_embedding`
- `keep_for_ambiance_embedding`
- `quality_score`
- `vector`
- `embedding_model`
- `embedding_dim`
- `embedding_version`
- `ingested_at`

Important note:
use cleaned images, not raw `social_images.csv`, as the main embedding input whenever Leo's cleaning pass is ready.

### Table: `menu_item_vectors`

Purpose:
dish-level semantic retrieval

Recommended row grain:
one row per parsed menu item

Recommended columns:

- `doc_id`
- `rest_id`
- `restaurant_name`
- `dish_name`
- `ingredients`
- `price`
- `text_for_embedding`
- `vector`
- `embedding_model`
- `embedding_dim`
- `embedding_version`
- `ingested_at`

Recommended embedding text:
`dish_name + ingredients + restaurant_name`

### Table: `review_vectors`

Purpose:
review retrieval, semantic backoff, review summarization, and restaurant-level aggregation

Recommended row grain:
one row per review

Recommended columns:

- `uid`
- `rest_id`
- `restaurant_name`
- `source`
- `text`
- `rating`
- `vector`
- `embedding_model`
- `embedding_dim`
- `embedding_version`
- `ingested_at`

Important note:
keep the original review text intact because Grace needs it for ABSA.

### Table: `restaurant_profiles`

Purpose:
one downstream-ready row per restaurant for analytics, clustering, prediction, and frontend summaries

Recommended columns:

- `rest_id`
- `restaurant_name`
- `bio_text`
- `bio_vector`
- `mean_food_image_vector`
- `mean_interior_image_vector`
- `mean_menu_vector`
- `mean_review_vector`
- `absa_food`
- `absa_service`
- `absa_ambiance`
- `image_count_clean`
- `interior_image_count`
- `review_count`
- `menu_item_count`
- `has_menu`
- `has_reviews`
- `has_food_images`
- `has_bio`

Optional phase-2 column:

- `fused_vector`

`fused_vector` should only be added after one of these is finalized:

- a shared projection space across modalities
- or a documented concatenation / weighting strategy for modality-specific aggregates

Until then, the modality-specific aggregate vectors should be treated as the source of truth.

## The Simplest Mental Model

For the team, the vector database should answer only four practical questions:

1. What menu items are most similar to this text query?
2. What food images are most similar to this image query?
3. What data do we currently have for restaurant `X`?
4. What is the current one-row summary for restaurant `X`?

Everything else should be built around those simple use cases.

## Recommended Team-Facing Usage Pattern

Downstream teammates should preferably interact with small helper functions, not raw DB internals.

The ideal usage pattern is:

1. Leo builds or refreshes the vector tables.
2. Leo exposes a few stable query helpers.
3. Everyone else uses those helpers or reads exported outputs.
4. Joins across modules always happen through `rest_id`.

That means most teammates do **not** need to know:

- how ANN indexing works
- how LanceDB stores files on disk
- how vectors are normalized internally

They only need to know:

- which table they should read from
- what key to join on
- what fields they can expect back

## What Each Table Is For In Plain English

### `image_vectors`

Think of this as:
"all cleaned restaurant images, one row per image, with an embedding attached"

Use this table when you want to:

- retrieve visually similar food photos
- choose representative food photos for a restaurant
- compute restaurant-level image averages later

### `menu_item_vectors`

Think of this as:
"all parsed dishes, one row per dish, with a text embedding attached"

Use this table when you want to:

- search for dishes by text
- find restaurants associated with a certain food idea
- power frontend dish search results

### `review_vectors`

Think of this as:
"all reviews, one row per review, with a text embedding attached"

Use this table when you want to:

- do semantic backoff when menu coverage is weak
- inspect review language for one restaurant
- group or summarize reviews

### `restaurant_profiles`

Think of this as:
"one final summary row per restaurant"

Use this table when you want to:

- build clustering inputs
- build prediction inputs
- render restaurant summary cards
- check what modalities are available for a restaurant

Do **not** use `restaurant_profiles` as the primary dish-search table.

## How Downstream Teammates Should Use The DB

### Merry (frontend)

Merry should think in terms of a user flow, not a DB flow.

For dish search and result cards:

- query `menu_item_vectors` first
- use `rest_id` to join restaurant metadata
- attach representative food images from `image_vectors`
- do not use `restaurant_profiles` as the primary dish search table

In practice, Merry's workflow should look like this:

1. user enters a text query such as "uni pasta" or "creative omakase dessert"
2. call a helper like `search_menu_items(query_text, top_k=20)`
3. get back matching dishes with `rest_id`
4. group or rank those results for display
5. fetch restaurant summaries or representative images using the returned `rest_id`s

For restaurant summary cards:

- read `restaurant_profiles`
- surface availability flags such as `has_menu`, `has_reviews`, and `has_food_images`
- gracefully handle restaurants that are missing one modality

Merry does not need to know how the vectors are stored internally. She mainly needs:

- stable result fields
- image paths
- restaurant names
- availability flags
- a score for ranking

### Grace (ABSA and integration)

Use:

- raw review text from `data/social_reviews.csv`
- or `review_vectors` if she also needs semantic grouping

Write back by `rest_id` into:

- `restaurant_profiles.absa_food`
- `restaurant_profiles.absa_service`
- `restaurant_profiles.absa_ambiance`

Grace should not depend on a cross-modal fused vector to begin her work.

In practice, Grace's workflow should look like this:

1. read all review rows for a restaurant or for the full dataset
2. run ABSA over raw review text
3. aggregate ABSA outputs by `rest_id`
4. write the final per-restaurant scores into `restaurant_profiles`

Grace mostly needs the DB as a stable storage and join layer, not as a search engine.

### Craig (clustering and prediction)

Use `restaurant_profiles` as the main modeling input table.

Recommended v1 feature strategy:

- start from `bio_vector`
- add `mean_food_image_vector`
- add `mean_menu_vector`
- add `mean_review_vector`
- add Grace's ABSA outputs
- concatenate or otherwise standardize offline in the modeling pipeline

Craig should not wait for a final `fused_vector` column if the modality-specific aggregates are already available.

In practice, Craig's workflow should look like this:

1. read `restaurant_profiles`
2. select the columns needed for clustering or regression
3. handle missing modalities explicitly
4. build model matrices offline in Python
5. write back cluster labels or prediction outputs by `rest_id` if needed

Craig should treat the DB as a clean feature source, not as the place where model training happens.

### Leo (vector DB owner)

Leo's priority should be:

1. finalize cleaned image metadata
2. generate per-table embeddings
3. build separated modality tables
4. expose minimal query functions for menu search, image search, and restaurant profile fetch

The first stable goal is not "full cross-modal magic." The first stable goal is a clean, queryable, auditable vector backend.

## Recommended Query Surface For The Team

Even if the storage backend is LanceDB, downstream teammates should consume a simple contract like this:

- `search_menu_items(query_text, top_k)`
- `search_food_images(query_image_or_vector, top_k)`
- `get_restaurant_profile(rest_id)`
- `get_restaurant_representatives(rest_id)`

If helpful, these functions can be understood in very plain terms:

- `search_menu_items(...)`
  - "give me the dishes most similar to this text"
- `search_food_images(...)`
  - "give me the images most similar to this image"
- `get_restaurant_profile(rest_id)`
  - "give me the one-row restaurant summary"
- `get_restaurant_representatives(rest_id)`
  - "give me the best images / dishes / review snippets to show for this restaurant"

Expected menu search output fields:

- `rest_id`
- `restaurant_name`
- `dish_name`
- `ingredients`
- `price`
- `score`
- `source_table`

Expected image search output fields:

- `rest_id`
- `restaurant_name`
- `image_uid`
- `image_path`
- `image_category`
- `score`
- `source_table`

Expected profile output fields:

- `rest_id`
- `restaurant_name`
- modality availability flags
- counts
- ABSA fields
- aggregate vectors if needed downstream

## Example Workflows

### Example 1: Merry builds a dish search page

User asks for:
"spicy seafood pasta"

Expected backend flow:

1. call `search_menu_items("spicy seafood pasta", top_k=20)`
2. receive top menu-item matches
3. collect the returned `rest_id`s
4. fetch representative image rows for those restaurants
5. render cards with:
   - restaurant name
   - dish name
   - ingredients
   - price
   - image
   - score

### Example 2: Grace writes ABSA outputs

Expected workflow:

1. read reviews grouped by `rest_id`
2. compute food/service/ambiance sentiment
3. aggregate to one score per restaurant per aspect
4. update `restaurant_profiles` using `rest_id`

Output shape should be conceptually like:

- one restaurant id
- one food sentiment score
- one service sentiment score
- one ambiance sentiment score

### Example 3: Craig prepares clustering inputs

Expected workflow:

1. read `restaurant_profiles`
2. filter to restaurants with enough available features
3. construct a matrix from vectors and ABSA features
4. run UMAP / clustering offline
5. optionally write cluster labels back by `rest_id`

## What Teammates Should Usually Not Do

- do not join tables on restaurant name if `rest_id` is available
- do not assume every restaurant has menu data
- do not assume every restaurant has a fully populated profile row on day one
- do not depend on internal LanceDB file layout
- do not hardcode old time-decay or trending fields from older drafts
- do not write directly to low-level vector tables unless that is your owned module

## Suggested Communication Rule

When teammates talk about the DB, use this language consistently:

- "search in `menu_item_vectors`"
- "fetch by `rest_id`"
- "read from `restaurant_profiles`"
- "write derived outputs back by `rest_id`"

This will keep the collaboration much simpler for teammates who are less familiar with database terminology.

## Known Constraints And Open Issues

### 1. Time-decay has been removed from scope

Time-decay appeared in older drafts and prototypes, but it is no longer part of the current implementation plan. Downstream teammates should ignore old references to decay weights, trending boosts, or timestamp-based ranking.

### 2. The current repo does not yet have a checked-in production DB build

There is no committed `vector_db/` state to hand off yet. Downstream teammates should rely on the schema contract, not on a local DB artifact.

### 3. Shared-space assumptions are still risky

Images, menus, reviews, and bios should not be mixed into one ANN table unless they are embedded into a validated shared space with the same dimension and retrieval behavior.

### 4. Cleaned image metadata is still a dependency

`social_images_cleaned.csv` is still the missing bridge between raw image scraping and a production-quality image vector table.

### 5. Partial restaurant coverage is normal right now

Some restaurants are missing menu rows. Frontend and modeling code must support incomplete feature coverage.

## Recommended Next-Step Order

1. Freeze the v1 schema contract above.
2. Finish `social_images_cleaned.csv`.
3. Generate `image_vectors`, `menu_item_vectors`, and `review_vectors`.
4. Build `restaurant_profiles` from those tables plus bios and ABSA outputs.
5. Expose small, stable retrieval functions for Merry and the rest of the team.
6. Revisit `fused_vector` only after the basic DB contract is stable.

## Bottom Line

Leo's main design idea is correct:

- item-level vectors for retrieval
- restaurant-level profiles for downstream analytics
- separated modality tables instead of one mixed table

The only real caution is that the current prototype and the current data are not yet ready for:

- mandatory timestamps
- mandatory shared cross-modal search through one mixed table
- mandatory `fused_vector`

If the team aligns on that narrower v1 contract, the vector database design is in a healthy place and downstream work can proceed without major schema churn.
