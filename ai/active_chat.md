## 1. **Edge Selection Strategy**

### Priority-Based Edge Creation
Which of these approaches do you prefer?

**Option A: Tiered Edge System**
- Tier 1 (always include): Targeted WR ↔ covering defenders, QB → all receivers, defenders within X yards of ball landing
- Tier 2 (conditionally include): Same-position matchups (WR-CB, TE-LB), nearby defenders
- Tier 3 (sparse sampling): Background context edges (distant players, same-team coordination)

**My thoughts** option A is best. I envision the following types of edges
EDGE_TYPES = {
    'qb_rr': QB → route runners,
    'qb_trr': QB → targeted route runners,
    'qb_def': QB -> coverage defender
    'rr_rr': route runner -> other route runner
    'rr_tr': route runner -> targeted route runner
    'rr_def': route runner -> coverage defender
    'trr_def': targeted route runner -> coverage defender
    'def_def': coverage defender -> coverage defender
}

### Specific Questions:
1. **Distance thresholds**: What's a reasonable max distance for "defender near ball landing"? 10 yards? 15?
    - we can experiment with different values. also we can experiment with different values relative to the time element. for example, the closer the ball is to arriving, the more it matters to be near that final landing spot. so what if we decay/decrease the distance_near_landing relative to the timeframe of the ball's path?
        *** decay_rate: rate the distance decays
        *** starting dist_near_landing:

2. **Targeted WR identification**: Should we use the `isTargeted` field from df_a? Or `targeted_defender` from coverage data?
    -- the isTargeted is for route runners. 
    -- the targeted_defender is for coverage defenders. 
    -- you could just make a unilateral in_tgt_relationship that uses both of those columns. would that be best?
3. **Coverage responsibility**: How reliable is the `coverage_responsibility` field? Should we trust it to define WR-DB matchups?
   -- coverage responsibility relates to the player's role relative to the scheme/coverage. so like curl flat defender, not to the player he is matched up on

---

## 2. **Edge Type Taxonomy**

### Proposed Edge Types
Based on the .md framework, I'm thinking:

```EDGE_TYPES = {
    'qb_rr': QB → route runners,
    'qb_trr': QB → targeted route runners,
    'qb_def': QB -> coverage defender
    'rr_rr': route runner -> other route runner
    'rr_tr': route runner -> targeted route runner
    'rr_def': route runner -> coverage defender
    'trr_def': targeted route runner -> coverage defender
    'def_def': coverage defender -> coverage defender
}
```

**Questions:**
1. Does this taxonomy make sense for your analysis goals?
2. Are there specific interactions you care about? 
-- targeted rr to def
-- qb and targeted rr


---

## 3. **Attention Prior (P_ij) Calculation**

### How to Compute Domain-Informed Priors
The .md mentions "log-prior term log P_ij(t)" - here's a proposed formula:

let's talk more about this after my answer/context provided with #1 and #2

**Questions:**
1. Does this weighting scheme align with your football intuition?
2. Should we make priors **time-varying**? (e.g., ball proximity becomes more important as ball approaches landing)
3. Do you want priors as a **soft weight** (continuous 0-1) or **hard mask** (include/exclude)?

---

## 4. **Temporal Edge Continuity**

### Tracking Edges Over Time
To support edgeRNNs, we need to track "this is the SAME edge evolving over time":

**Proposed approach:**
```python
# Add edge_id column that persists across frames
edge_id = f"{game_id}_{play_id}_{min(playerA_id, playerB_id)}_{max(playerA_id, playerB_id)}"

# Then edges can be grouped by edge_id and sorted by frame_id for temporal modeling
```

**Questions:**
1. Should edge IDs be undirected (A↔B treated as one edge) or directed (A→B and B→A are different)?
-- it would def be easier computationally and overall to treat it as undirected. but what do you think? is the complexity worth it?
2. Do we need to handle cases where an edge "appears" mid-play (e.g., defender rotates into coverage)?
-- im not sure, but i'd imagine edges should evolve overtime. since we are learning our edges through the attention mechanism and attention-based adjacency matrix though, this will automatically be captured right?

---

## 5. **Ball Trajectory Features**

### Current vs. Enhanced
Current script has: `ball_x_t, ball_y_t, ball_progress, frames_to_landing`

**Should we add:**
- **Ball velocity vector** at frame t (interpolated from trajectory)?
- **Hang time remaining** (seconds, not just frames)?
- **Catch probability at time t** (if you have a pre-trained model or heuristic)?
- **Pursuit angle quality** (is defender running toward optimal intercept point)?

**Question:** Which of these would be useful for your eventual defender impact analysis?
- ignore the hang time and ball velocity. as for catch probability, we don't really have much besides cp from nflfastR. that model is without player trackign data so its better as just a heuristic
- pursuit angle idk, lets just ignore and skip it for now. i think we cover it in a different way

---

## 6. **Player Role Granularity**

### Current Coverage Context
The script includes `coverage_responsibility`, `coverage_scheme`, but doesn't use them to structure edges.

**Questions:**
1. Should we create **different edge types** for different coverage schemes? (e.g., man-coverage edges in Cover-1 vs. zone responsibilities in Cover-3)
2. Do you want **alignment-based edges**? (e.g., slot CB to slot WR is more important than outside CB to slot WR)
3. Should we differentiate **pre-snap vs. post-snap** edges? (Initial matchups vs. how they evolve)

- since we are learning edges through attention and the attention adjacency matrix, if we include coverage_responsibility and coverage_scheme in the training features, then won't the model learn how to adjust edges based on these?
- alignment doesn't really matter
- again, the data is only from when the ball is in the air. so there is no post-snap vs pre-snap edges. although, we have 2 types of player tracking data. input (before ball is landed and caught) and output (after). the output data only has play_id, game_id, nfl_id, frame_id, x, y though. 

---

## 7. **Output Format & Compatibility**

### Backward Compatibility
**Questions:**
1. Should the new script produce a **different version** (e.g., `dataframe_c/v2.parquet`) or **replace v1**?
2. Do you want to keep the old fully-connected version available for comparison?
3. Should the output schema be **backward compatible** (same columns + new ones) or can it be a breaking change?

-- let's do a different version. so v2
-- i will archive old versions of scripts and data to use if needed
-- it can be breaking change. we are rebuilding everything after the data processing. we are just modify df_c to match that and better enable us. so yes, everything downstream will be new and we have freedom

---

## 8. **Computational Constraints**

### Edge Count Explosion
Fully connected: ~22 players → 462 edges per frame
Filtered approach: potentially 50-100 edges per frame (much more manageable)

**Questions:**
1. Are you okay with **non-deterministic edge counts** per frame? (Some frames might have 60 edges, others 120)
-- yes. different numbers of players and frames in each play. it can't be deterministic
2. Should we **guarantee a minimum edge count** (e.g., always include top-K closest pairs as fallback)?
-- no
3. Any memory/storage constraints I should be aware of?
-- no, but try to be as computationally efficient as possible

---

## 9. **Validation & Testing**

### How to Verify It's Working
**Questions:**
1. Should the script output **diagnostic visualizations**? (e.g., "% of edges by type", "average edges per frame")
2. Do you want a **sample play visualization** showing which edges were created and why?
3. Should we log cases where **expected edges are missing**? (e.g., "Warning: Targeted WR has no covering defender edge")

- yes to all that. create a new file path for that. 

---

## 10. **Integration with Training Scripts**

### Downstream Compatibility
The current `train_completion.py` and `train_yac_epa.py` expect certain edge features.

**Questions:**
1. Should the new edge features be a **superset** of the old ones (add columns, don't remove)?
2. Do we need to update the training scripts to **use edge types** in the attention mechanism?
3. Or should this script be **training-script agnostic** (just produce better edges, let training decide how to use them)?

-- use the existing stuff as inspiration, but downstream everything is unfinished so we have some flexibility and freedom.
---

## **My Suggested Prioritization**

If you want to answer just the **critical** questions first:

1. **Edge selection approach** (Option A/B/C above)
2. **Edge type taxonomy** (which types matter most?)
3. **Attention prior format** (soft weights vs. hard masks)
4. **Temporal continuity** (do we need edge_id tracking now, or defer to later training script updates?)

The rest can use reasonable defaults that we refine later.

**What do you think? Which questions can you answer now, and which need more investigation?**