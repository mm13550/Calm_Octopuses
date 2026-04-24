# Vector Database Handoff For Downstream Teammates

## Read This First

This is the current downstream-facing handoff for the vector database.

- the older project PDF is **not** the current source of truth
- `time-decay` has been removed from scope
- LanceDB is being built following Leo's handoff, but the full DB is still in progress

If you only remember three things, remember these:

1. `rest_id` is the main key used to connect everything.
2. The project has both **raw data files** and **vector DB tables**.
3. Most teammates will only use a small part of the DB, not the whole system.

## 1. Current Data Configuration

These are the main upstream data files that already exist in the repo.

| File | What one row means | Size | Main fields | Who will likely care |
|---|---|---:|---|---|
| `data/csv/restaurant_lookup.csv` | one restaurant | 349 rows | `name`, `rest_id`, `homepage`, join metadata | everyone |
| `data/social_reviews.csv` | one review | 1,742 rows | `uid`, `rest_id`, `source`, `text`, `rating` | Grace, Leo |
| `data/social_images.csv` | one image | 13,778 rows | `image_uid`, `rest_id`, `source`, `image_path` | Leo, Merry |
| `data/extracted_menus/final_parsed_menus.json` | one menu item / dish | 9,261 rows | `rest_id`, `restaurant_name`, `dish_name`, `ingredients`, `price` | Merry, Leo |
| `data/extracted_bios/restaurant_bios_joinable.json` | one restaurant bio | 349 rows | `rest_id`, `name`, `bio` | Craig, Grace, Merry |

### Important coverage note

- reviews, images, and bios currently cover 349 restaurants
- menus currently cover 310 restaurants
- partial coverage is normal right now

So if your module expects every restaurant to have menu data, it should handle missing rows gracefully.

## 2. How To Think About The System

The simplest way to think about the project is:

### Raw data layer

- `restaurant_lookup.csv` tells us **which restaurant is which**
- `social_reviews.csv` stores raw review text
- `social_images.csv` stores raw image metadata and image paths
- `final_parsed_menus.json` stores parsed dishes
- `restaurant_bios_joinable.json` stores restaurant-level bio text

### Vector DB layer

Leo is turning those raw files into searchable vector tables in LanceDB.

### Downstream layer

Different teammates read different tables depending on their task:

- search / frontend
- ABSA
- clustering / prediction

## 3. Current LanceDB Structure (In Progress)

Think of these as the main tables the team will read from once the DB build is ready.

| Table | What one row means | Main contents | Who will probably use it |
|---|---|---|---|
| `menu_item_vectors` | one menu item | dish text + embedding + restaurant metadata | Merry, Leo |
| `image_vectors` | one cleaned image | image path + image metadata + embedding | Merry, Leo |
| `review_vectors` | one review | review text + rating + embedding | Grace, Leo |
| `restaurant_profiles` | one restaurant | aggregated restaurant-level features | Craig, Grace, Merry |

### What each table is for

`menu_item_vectors`

- use this for dish search
- example: "show me dishes similar to uni pasta"

`image_vectors`

- use this for food-image retrieval and representative restaurant images
- example: "show me the best food photos for this restaurant"

`review_vectors`

- use this for review-level text analysis or semantic backoff
- example: "pull review text related to this restaurant or concept"

`restaurant_profiles`

- use this for one-row-per-restaurant downstream features
- example: clustering, prediction inputs, restaurant summary cards

## 4. The Main Join Rule

Use `rest_id` for joins.

That means:

- join raw files by `rest_id`
- join vector tables by `rest_id`
- write derived outputs back by `rest_id`

Do **not** join by restaurant name if `rest_id` is available.

## 5. What You Will Probably Use

### Merry

Merry will probably use:

- `menu_item_vectors`
- `image_vectors`
- `restaurant_profiles`

Most likely tasks:

- dish search
- result cards
- restaurant summary cards
- attaching representative food images to search results

### Grace

Grace will probably use:

- `data/social_reviews.csv`
- `review_vectors`
- `restaurant_profiles`

Most likely tasks:

- ABSA on raw review text
- grouping review outputs by `rest_id`
- writing restaurant-level sentiment outputs back into `restaurant_profiles`

### Craig

Craig will probably use:

- `restaurant_profiles`
- restaurant-level bio / image / menu / review aggregates

Most likely tasks:

- clustering inputs
- prediction inputs
- feature matrix construction

### Leo

Leo will probably use:

- all vector tables
- cleaned image metadata
- ingestion / refresh scripts

Most likely tasks:

- build vectors
- ingest vectors into LanceDB
- expose search helpers for the rest of the team

## 6. Common Workflows

### Workflow A: Dish search for frontend

1. Search `menu_item_vectors` with a text query.
2. Get back matching dishes and their `rest_id`s.
3. Use those `rest_id`s to fetch restaurant info and representative images.
4. Render the results in the UI.

### Workflow B: ABSA

1. Read review text from `data/social_reviews.csv` or `review_vectors`.
2. Run ABSA grouped by `rest_id`.
3. Write final restaurant-level scores back to `restaurant_profiles`.

### Workflow C: Clustering / prediction

1. Read `restaurant_profiles`.
2. Build a modeling matrix from restaurant-level features.
3. Train models offline in Python.
4. Optionally write outputs back by `rest_id`.

## 7. What Is Inside `restaurant_profiles`

This is the table most downstream teammates should think of as the final restaurant-level summary table.

Expected fields include:

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

This is the table Craig will likely use most.
Grace will likely write part of it.
Merry will likely read it for restaurant cards and metadata.

## 8. What Is Not Part Of The Current Scope

These older ideas should **not** be treated as current requirements:

- time-decay
- trending score logic
- timestamp-based retrieval ranking

Also, do not assume:

- every restaurant has menu data
- every teammate needs to touch LanceDB directly
- the current prototype in `algorithms/retrieval_engine.py` is the final team-facing API

## 9. Practical Rules

- Start with `restaurant_lookup.csv` if you need stable restaurant identity.
- Use `rest_id` everywhere.
- Use item-level tables for search.
- Use `restaurant_profiles` for restaurant-level modeling and summaries.
- Handle missing menu coverage gracefully.
- If you are unsure which table to use, ask: "Am I searching for items, or am I reading one row per restaurant?"

That rule alone will answer most table-selection questions.
