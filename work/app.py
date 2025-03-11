import streamlit as st
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import HtmlFormatter
from streamlit.components.v1 import html

# 必须作为第一个Streamlit命令
st.set_page_config(layout="wide")


def load_styles():
    """加载CSS样式"""
    try:
        with open("style.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("样式文件加载失败")


def init_session():
    """初始化session状态"""
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
    """验证并清洗图数据"""
    cleaned = {"nodes": [], "edges": []}
    seen_ids = set()

    # 处理节点
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

    # 处理边
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
    """带高亮的JSON编辑器组件"""
    formatter = HtmlFormatter(style="colorful", full=True, cssclass="highlight")

    col1, col2 = st.columns([1, 1])
    with col1:
        new_json = st.text_area(
            "编辑 JSON 内容",
            value=st.session_state.raw_json,
            height=600,
            key=f"editor_{st.session_state.editor_key}",
            help="修改后点击保存按钮更新图表"
        )

        if st.button("💾 保存修改", type="primary"):
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
                st.error(f"保存失败: {str(e)}")

    with col2:
        st.markdown("**实时预览**")
        if st.session_state.raw_json:
            html_code = highlight(
                st.session_state.raw_json,
                JsonLexer(),
                formatter
            )
            html(f'<div class="highlight">{html_code}</div>', height=600, scrolling=True)


def build_network_graph(data):
    """构建带权重的NetworkX图"""
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
    """生成带高亮路径的网络图"""
    try:
        G = build_network_graph(data)
        pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(len(G.nodes)), iterations=200)

        fig, ax = plt.subplots(figsize=(12, 9), dpi=120)

        # 绘制基础元素
        nx.draw_networkx_nodes(
            G, pos,
            node_size=[n["size"] for n in data["nodes"]],
            node_color=[n["color"] for n in data["nodes"]],
            edgecolors="white",
            linewidths=1.5,
            ax=ax
        )

        # 绘制普通边
        nx.draw_networkx_edges(
            G, pos,
            edge_color=[e["color"] for e in data["edges"]],
            width=[e["width"] * 0.7 for e in data["edges"]],
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            ax=ax
        )

        # 高亮最短路径
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

        # 标签处理
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
        st.error(f"图表生成失败: {str(e)}")
        return None


def main():
    load_styles()
    init_session()

    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🌐 智能网络路径分析系统</h1>",
                unsafe_allow_html=True)

    # 文件上传
    uploaded_file = st.file_uploader("上传JSON文件", type=["json"], key="uploaded_file")

    # 处理文件上传
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
            st.error(f"文件处理失败: {str(e)}")

    # 三列布局（确保在main函数顶层定义）
    col1, col2, col3 = st.columns([2.5, 4, 1.5])  # 这是关键修复点

    with col1:
        st.header("数据输入")
        if st.session_state.raw_json:
            json_editor_component()
        else:
            st.info("请上传JSON文件")

    with col2:
        st.header("可视化展示")
        if st.session_state.json_data:
            fig = generate_network_graph(st.session_state.json_data)
            if fig:
                st.pyplot(fig)
        else:
            st.info("等待数据输入...")

    with col3:  # 确保col3在此作用域内定义
        st.header("路径分析")

        # 输入节点
        start_node = st.text_input("起始节点", key="path_start")
        end_node = st.text_input("目标节点", key="path_end")

        # 查找按钮
        if st.button("🔍 分析路径", type="primary", use_container_width=True):
            st.session_state.show_paths = True
            try:
                G = build_network_graph(st.session_state.json_data)

                # 计算最短路径
                shortest_path = nx.shortest_path(G, start_node, end_node, weight="weight")
                st.session_state.shortest_path = shortest_path

                # 计算权重
                total_weight = sum(G.edges[u, v]['weight']
                                   for u, v in zip(shortest_path[:-1], shortest_path[1:]))
                st.session_state.total_weight = round(total_weight, 2)

                # 查找所有路径
                all_paths = list(nx.all_simple_paths(G, start_node, end_node, cutoff=5))
                st.session_state.found_paths = all_paths[:5]

            except Exception as e:
                st.error(str(e))

        # 显示结果
        if st.session_state.show_paths:
            st.subheader("分析结果")

            if st.session_state.shortest_path:
                st.success(f"""
                    ​**最短路径**  
                    {' → '.join(st.session_state.shortest_path)}  
                    总权重: {st.session_state.total_weight}
                """)

            if st.session_state.found_paths:
                with st.expander("所有路径"):
                    for i, path in enumerate(st.session_state.found_paths, 1):
                        st.code(f"路径{i}: {' → '.join(path)}")


if __name__ == "__main__":
    main()