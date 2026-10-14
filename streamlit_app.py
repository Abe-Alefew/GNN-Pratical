"""Streamlit client for the containerized GNN prediction API."""

import json
import random
from typing import Any

import httpx
import streamlit as st


def api_request(
    base_url: str,
    method: str,
    path: str,
    payload: Any = None,
    timeout_seconds: float = 15.0,
) -> tuple[int, Any]:
    """Call the API and return the status code and decoded response."""
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.request(method, url, json=payload)
    except httpx.HTTPError as error:
        return 0, {"detail": f"Could not connect to API: {error}"}

    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text or "API returned an empty response"}
    return response.status_code, body


def parse_json(label: str, value: str) -> Any:
    """Parse a JSON editor value and show a useful UI error on failure."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        st.error(f"{label} is not valid JSON: {error.msg}")
        return None


def build_request(
    node_count: int,
    feature_count: int,
    feature_mode: str,
    feature_scale: float,
    seed: int,
    topology: str,
    target_node_ids: list[int],
) -> dict[str, Any]:
    """Generate a prediction payload from compact UI settings."""
    random_generator = random.Random(seed)
    nodes = []
    for node_id in range(node_count):
        if feature_mode == "Deterministic random":
            features = [
                round(random_generator.uniform(-feature_scale, feature_scale), 6)
                for _ in range(feature_count)
            ]
        elif feature_mode == "One-hot probe":
            features = [0.0] * feature_count
            features[node_id % feature_count] = feature_scale
        else:
            features = [0.0] * feature_count
        nodes.append({"node_id": node_id, "features": features})

    if topology == "Fully connected":
        edges = [
            [source, target]
            for source in range(node_count)
            for target in range(node_count)
            if source != target
        ]
    elif topology == "Self-loops only":
        edges = [[node_id, node_id] for node_id in range(node_count)]
    else:
        edges = [
            edge
            for node_id in range(node_count - 1)
            for edge in ([node_id, node_id + 1], [node_id + 1, node_id])
        ]

    return {
        "nodes": nodes,
        "edges": edges,
        "target_node_ids": target_node_ids,
    }


st.set_page_config(
    page_title="GNN Inference Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #081f29; color: #e8f4f1; }
    [data-testid="stHeader"] { background: #081f29; }
    [data-testid="stToolbar"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1400px; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    label { color: #d8ebe7 !important; }
    [data-testid="stMetric"] { background: #123842; border: 1px solid #255864;
                                padding: .75rem; border-radius: 8px; }
    [data-testid="stMetricLabel"] { color: #9fc9c1 !important; }
    [data-testid="stMetricValue"] { color: #f1fbf8 !important; }
    [data-baseweb="textarea"], [data-baseweb="input"] { background: #123842; }
    textarea, input { color: #f1fbf8 !important; background: #123842 !important; }
    textarea::placeholder, input::placeholder { color: #8fb2ad !important; }
    [data-testid="stExpander"] { background: #0e3039; border-color: #255864; }
    [data-testid="stSidebar"] { background: #102a2e; }
    [data-testid="stSidebar"] * { color: #edf7f4; }
    .hero { padding: 1.4rem 1.6rem; background: #103f43; color: #f5fbf8;
            border-radius: 12px; margin-bottom: 1.2rem; }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p { margin: .4rem 0 0; color: #b9ded4; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<section class="hero"><h1>GNN Inference Console</h1>'
    '<p>Send graph context to the Docker-hosted prediction service.</p></section>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connection")
    base_url = st.text_input("API base URL", value="http://localhost:8000")
    timeout_seconds = st.slider("Request timeout (seconds)", 5, 120, 15)
    check_connection = st.button("Check connection", use_container_width=True, type="primary")

    if check_connection:
        health_status, health_body = api_request(
            base_url, "GET", "/health", timeout_seconds=timeout_seconds
        )
        if health_status == 200:
            st.success(f"Healthy · model {health_body.get('model_version', 'unknown')}")
        else:
            st.error(f"API unavailable ({health_status or 'connection error'})")
            st.json(health_body)

    st.divider()
    st.caption("Start the API with:")
    st.code("docker run --rm -p 8000:8000 gnn-mlops-api:latest", language="bash")

model_status, model_body = api_request(
    base_url, "GET", "/model-info", timeout_seconds=timeout_seconds
)
feature_count = None
if model_status == 200:
    feature_count = model_body.get("num_features")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Model", model_body.get("model_type", "unknown"))
    metric_columns[1].metric("Features", model_body.get("num_features", "-"))
    metric_columns[2].metric("Classes", model_body.get("num_classes", "-"))
    metric_columns[3].metric("Layers", model_body.get("num_layers", "-"))
else:
    st.info("Connect the API to load model metadata. You can still prepare a request below.")

st.subheader("Prediction builder")
st.caption("Configure a graph below. The request payload is generated automatically.")

if feature_count:
    config_columns = st.columns(4)
    with config_columns[0]:
        node_count = st.number_input("Nodes", min_value=2, max_value=100, value=2, step=1)
    with config_columns[1]:
        feature_mode = st.selectbox(
            "Feature strategy",
            ["Zero vector", "One-hot probe", "Deterministic random"],
            help="These are generated test features; real Cora features give meaningful predictions.",
        )
    with config_columns[2]:
        feature_scale = st.number_input("Feature scale", min_value=0.01, max_value=10.0, value=1.0)
    with config_columns[3]:
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    graph_columns = st.columns(2)
    with graph_columns[0]:
        topology = st.selectbox(
            "Graph topology",
            ["Bidirectional chain", "Fully connected", "Self-loops only"],
        )
    with graph_columns[1]:
        target_node_ids = st.multiselect(
            "Target nodes",
            options=list(range(int(node_count))),
            default=list(range(min(2, int(node_count)))),
        )

    request_body = build_request(
        node_count=int(node_count),
        feature_count=int(feature_count),
        feature_mode=feature_mode,
        feature_scale=float(feature_scale),
        seed=int(seed),
        topology=topology,
        target_node_ids=target_node_ids,
    )
    st.caption(
        f"Generated graph: {len(request_body['nodes'])} nodes, "
        f"{len(request_body['edges'])} edges, {len(target_node_ids)} targets, "
        f"{feature_count} features per node."
    )

    send_request = st.button("Run prediction", type="primary", use_container_width=True)

    if send_request:
        if not target_node_ids:
            st.error("Select at least one target node.")
            st.stop()
        response_status, response_body = api_request(
            base_url,
            "POST",
            "/predict",
            request_body,
            timeout_seconds=timeout_seconds,
        )
        st.session_state["last_request"] = request_body
        st.session_state["last_response"] = response_body
        st.session_state["last_status"] = response_status

    with st.expander("Advanced: inspect generated request JSON"):
        st.json(request_body)
else:
    st.warning("Connect to the API to load the model feature count and enable the request builder.")

if "last_response" in st.session_state:
    st.divider()
    response_status = st.session_state["last_status"]
    if response_status == 200:
        st.success(f"Prediction completed · HTTP {response_status}")
    else:
        st.error(f"Prediction failed · HTTP {response_status or 'connection error'}")

    result_columns = st.columns(2)
    with result_columns[0]:
        st.subheader("Predictions")
        predictions = st.session_state["last_response"].get("predictions", [])
        if predictions:
            st.dataframe(predictions, use_container_width=True, hide_index=True)
        else:
            st.json(st.session_state["last_response"])
    with result_columns[1]:
        st.subheader("Response JSON")
        st.json(st.session_state["last_response"])

    with st.expander("Request JSON"):
        st.json(st.session_state["last_request"])