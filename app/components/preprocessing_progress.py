import streamlit as st


STEP_KEYS = [
    "overview",
    "understanding",
    "cleaning",
    "engineering",
    "transformation",
]

STEP_META = {
    "overview": {
        "title": "Data Overview",
        "subtitle": "Load your dataset and get a high-level summary",
        "number": 1,
    },
    "understanding": {
        "title": "Data Understanding",
        "subtitle": "Distributions, correlations, and feature statistics",
        "number": 2,
    },
    "cleaning": {
        "title": "Data Cleaning",
        "subtitle": "Handle missing values and fix incorrect data types",
        "number": 3,
    },
    "engineering": {
        "title": "Feature Engineering & Selection",
        "subtitle": "Outlier removal and categorical encoding",
        "number": 4,
    },
    "transformation": {
        "title": "Data Transformation",
        "subtitle": "Scaling, normalisation, standardisation, power transforms",
        "number": 5,
    },
}


def init_preprocessing_progress():
    if "prep_progress" not in st.session_state:
        st.session_state["prep_progress"] = {k: False for k in STEP_KEYS}

    if "prep_open_step" not in st.session_state:
        st.session_state["prep_open_step"] = "overview"


def is_step_unlocked(step_key: str) -> bool:
    idx = STEP_KEYS.index(step_key)
    if idx == 0:
        return True
    prev_key = STEP_KEYS[idx - 1]
    return st.session_state["prep_progress"].get(prev_key, False)


def mark_step_complete(step_key: str):
    st.session_state["prep_progress"][step_key] = True
    idx = STEP_KEYS.index(step_key)

    # Move the open expander to the next step
    if idx + 1 < len(STEP_KEYS):
        st.session_state["prep_open_step"] = STEP_KEYS[idx + 1]
    else:
        st.session_state["prep_open_step"] = None

def is_step_open(step_key: str) -> bool:
    return st.session_state.get("prep_open_step") == step_key


def get_completed_count() -> int:
    return sum(st.session_state["prep_progress"].values())


def render_progress_tracker():
    completed = get_completed_count()
    total = len(STEP_KEYS)

    st.markdown("### Pipeline Progress")
    st.progress(completed / total, text=f"{completed}/{total} steps completed")

    cols = st.columns(len(STEP_KEYS))
    for col, key in zip(cols, STEP_KEYS):
        meta = STEP_META[key]
        unlocked = is_step_unlocked(key)
        done = st.session_state["prep_progress"].get(key, False)

        if done:
            label = f"✅ {meta['number']}. {meta['title']}"
        elif unlocked:
            label = f"🔵 {meta['number']}. {meta['title']}"
        else:
            label = f"🔒 {meta['number']}. {meta['title']}"

        col.markdown(label)


def render_step_header(step_key: str):
    meta = STEP_META[step_key]
    done = st.session_state["prep_progress"].get(step_key, False)
    unlocked = is_step_unlocked(step_key)

    badge = "✅ Complete" if done else "Active" if unlocked else "Locked"
    st.markdown(
        f"""
### Step {meta['number']} · {meta['title']}
{meta['subtitle']}

**Status:** {badge}
"""
    )


def render_complete_button(step_key: str, label: str = "Mark Step Complete & Continue"):
    if st.button(label, key=f"complete_{step_key}"):
        mark_step_complete(step_key)
        st.rerun()