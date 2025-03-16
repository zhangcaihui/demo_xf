import streamlit as st
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import HtmlFormatter
from streamlit.components.v1 import html
import time

# Must be the first Streamlit command
st.set_page_config(layout="wide")

from streamlit.components.v1 import html

def load_styles():
    """Load CSS styles"""
    try:
        with open("style.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Failed to load style file")


def init_session():
    """Initialize session state"""
    session_keys = {
        "json_data": None,
        "raw_json": "",
        "editor_key": 0,
        "file_processed": False,
        "show_paths": False,
        "found_paths": [],
        "shortest_path": [],
        "total_weight": 0.0
    }
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = session_keys[key]


def validate_graph_data(data):
    """Validate and clean graph data"""
    cleaned = {"nodes": [], "edges": []}
    seen_ids = set()

    # Process nodes
    for node in data.get("nodes", []):
        if isinstance(node, dict) and "id" in node:
            node_id = str(node["id"])
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                cleaned["nodes"].append({
                    "id": node_id,
                    "label": node.get("label", node_id),
                    "color": node.get("color", "#4B8BBE"),
                    "size": max(100, min(2000, node.get("size", 800)))
                })

    # Process edges
    valid_nodes = {n["id"] for n in cleaned["nodes"]}
    for edge in data.get("edges", []):
        if isinstance(edge, dict):
            src = str(edge.get("source", ""))
            tgt = str(edge.get("target", ""))
            if src in valid_nodes and tgt in valid_nodes:
                cleaned["edges"].append({
                    "source": src,
                    "target": tgt,
                    "color": edge.get("color", "#666666"),
                    "width": max(0.5, min(5.0, edge.get("width", 1.5))),
                    "arrowstyle": edge.get("arrowstyle", "->")
                })
    return cleaned


def json_editor_component():
    """JSON editor component with syntax highlighting"""
    formatter = HtmlFormatter(style="colorful", full=True, cssclass="highlight")

    col1, col2 = st.columns([1, 1])
    with col1:
        new_json = st.text_area(
            "Edit JSON Content",
            value=st.session_state.raw_json,
            height=600,
            key=f"editor_{st.session_state.editor_key}",
            help="Click save button after modification to update the chart"
        )

        if st.button("💾 Save Changes", type="primary"):
            try:
                if "uploaded_file" in st.session_state:
                    del st.session_state.uploaded_file
                    st.session_state.file_processed = False

                parsed = json.loads(new_json)
                st.session_state.json_data = validate_graph_data(parsed)
                st.session_state.raw_json = json.dumps(
                    st.session_state.json_data,
                    indent=2,
                    ensure_ascii=False
                )
                st.session_state.editor_key += 1
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {str(e)}")

    with col2:
        st.markdown("**Live Preview**")
        if st.session_state.raw_json:
            html_code = highlight(
                st.session_state.raw_json,
                JsonLexer(),
                formatter
            )
            html(f'<div class="highlight">{html_code}</div>', height=600, scrolling=True)


def build_network_graph(data):
    """Build NetworkX graph with weights"""
    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"])
    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge.get("width", 1.0),
            color=edge.get("color", "#666666")
        )
    return G


def generate_network_graph(data):
    """Generate network graph with highlighted paths"""
    try:
        G = build_network_graph(data)
        pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(len(G.nodes)), iterations=200)

        fig, ax = plt.subplots(figsize=(12, 9), dpi=120)

        # Draw basic elements
        nx.draw_networkx_nodes(
            G, pos,
            node_size=[n["size"] for n in data["nodes"]],
            node_color=[n["color"] for n in data["nodes"]],
            edgecolors="white",
            linewidths=1.5,
            ax=ax
        )

        # Draw regular edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=[e["color"] for e in data["edges"]],
            width=[e["width"] * 0.7 for e in data["edges"]],
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            ax=ax
        )

        # Highlight shortest path
        if st.session_state.shortest_path:
            path_edges = list(zip(st.session_state.shortest_path[:-1],
                                  st.session_state.shortest_path[1:]))
            nx.draw_networkx_edges(
                G, pos,
                edgelist=path_edges,
                edge_color="#FF4444",
                width=3.5,
                arrowsize=25,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.2",
                ax=ax
            )
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=st.session_state.shortest_path,
                node_color="#FFD700",
                edgecolors="#FF4444",
                linewidths=2,
                ax=ax
            )

        # Label processing
        labels = {n["id"]: n.get("label", "") for n in data["nodes"]}
        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=10,
            font_family='Microsoft YaHei',
            bbox=dict(facecolor='white', alpha=0.8),
            ax=ax
        )

        plt.axis("off")
        return fig
    except Exception as e:
        st.error(f"Chart generation failed: {str(e)}")
        return None


def main():

    load_styles()
    init_session()

    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🌐 Intelligent Network Path Analysis System</h1>",
                unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader("Upload JSON File", type=["json"], key="uploaded_file")

    # Process file upload
    if uploaded_file and not st.session_state.file_processed:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            raw_data = json.loads(content)
            st.session_state.json_data = validate_graph_data(raw_data)
            st.session_state.raw_json = json.dumps(
                st.session_state.json_data,
                indent=2,
                ensure_ascii=False
            )
            st.session_state.file_processed = True
        except Exception as e:
            st.error(f"File processing failed: {str(e)}")

    # Three-column layout
    col1, col2, col3 = st.columns([2.5, 4, 1.5])

    with col1:

        st.header("Data Input")
        if st.session_state.raw_json:
            json_editor_component()
        else:
            st.info("Please upload a JSON file")

    with col2:

        st.header("Visualization")
        if st.session_state.json_data:
            fig = generate_network_graph(st.session_state.json_data)
            if fig:
                st.pyplot(fig)
        else:
            st.info("Waiting for data input...")

    with col3:

        st.header("Path Analysis")

        # Node inputs
        start_node = st.text_input("Start Node", key="path_start")
        end_node = st.text_input("End Node", key="path_end")

        # Analysis button
        if st.button("🔍 Analyze Path", type="primary", use_container_width=True):
            st.session_state.show_paths = True
            try:
                G = build_network_graph(st.session_state.json_data)

                # Calculate shortest path
                shortest_path = nx.shortest_path(G, start_node, end_node, weight="weight")
                st.session_state.shortest_path = shortest_path

                # Calculate total weight
                total_weight = sum(G.edges[u, v]['weight']
                                   for u, v in zip(shortest_path[:-1], shortest_path[1:]))
                st.session_state.total_weight = round(total_weight, 2)

                # Find all paths
                all_paths = list(nx.all_simple_paths(G, start_node, end_node, cutoff=5))
                st.session_state.found_paths = all_paths[:5]

            except Exception as e:
                st.error(str(e))

        # Display results
        if st.session_state.show_paths:
            st.subheader("Analysis Results")

            if st.session_state.shortest_path:
                st.success(f"""
                    ​**Shortest Path**  
                    {' → '.join(st.session_state.shortest_path)}  
                    Total weight: {st.session_state.total_weight}
                """)

            if st.session_state.found_paths:
                with st.expander("All Paths"):
                    for i, path in enumerate(st.session_state.found_paths, 1):
                        st.code(f"Path {i}: {' → '.join(path)}")


if __name__ == "__main__":
    main()