import streamlit as st
import json
import networkx as nx
import matplotlib.pyplot as plt
import random
import os
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import HtmlFormatter
from streamlit.components.v1 import html
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch

# 初始化页面配置
st.set_page_config(layout="wide", page_title="机器人能力事件分析系统")

# 自定义CSS样式
CSS = """
<style>
.highlight {
    border-radius: 0.5rem;
    padding: 1rem !important;
    margin-top: 1rem;
    background: #f8f9fa !important;
}
.node-capability {
    fill: #4B8BBE;
    stroke: #2E5D87;
}
.node-provider {
    fill: #FFA500;
    stroke: #CC8400;
}
.node-self-implemented {
    fill: #9b59b6;
    stroke: #8e44ad;
}
.edge-early {
    stroke: #2ecc71 !important;
}
.edge-late {
    stroke: #e74c3c !important;
}
.stTab > div[role='tablist'] {
    margin-bottom: 1rem;
}
.path-error {
    background-color: #fff3f3;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff4b4b;
}
.node-tooltip {
    position: absolute;
    background: rgba(255, 255, 255, 0.9);
    padding: 5px;
    border: 1px solid #ddd;
    border-radius: 3px;
    pointer-events: none;
    font-size: 12px;
}
.mapping-table {
    font-size: 0.8rem;
    max-height: 300px;
    overflow-y: auto;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


class NodeMapper:
    def __init__(self):
        self.real_to_mapped = {}
        self.mapped_to_real = {}
        self.next_id = 0
        self.mapping_file = "node_mapping.json"
        self.load_mapping()

    def load_mapping(self):
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r') as f:
                data = json.load(f)
                self.real_to_mapped = data.get("real_to_mapped", {})
                self.mapped_to_real = data.get("mapped_to_real", {})
                self.next_id = data.get("next_id", 0)

    def save_mapping(self):
        with open(self.mapping_file, 'w') as f:
            json.dump({
                "real_to_mapped": self.real_to_mapped,
                "mapped_to_real": self.mapped_to_real,
                "next_id": self.next_id
            }, f, indent=2)

    def map_node(self, real_name):
        if real_name not in self.real_to_mapped:
            mapped_id = self._generate_mapped_id()
            self.real_to_mapped[real_name] = mapped_id
            self.mapped_to_real[mapped_id] = real_name
            self.next_id += 1
            self.save_mapping()
        return self.real_to_mapped[real_name]

    def _generate_mapped_id(self):
        if self.next_id < 26:
            return chr(ord('A') + self.next_id)
        else:
            prefix = chr(ord('A') + (self.next_id // 26 - 1))
            suffix = chr(ord('A') + (self.next_id % 26))
            return f"{prefix}{suffix}"

    def get_real_name(self, mapped_id):
        return self.mapped_to_real.get(mapped_id, mapped_id)

    def get_node_type(self, mapped_id):
        real_name = self.get_real_name(mapped_id)
        return real_name.split(':')[0] if ':' in real_name else 'unknown'


node_mapper = NodeMapper()


def init_session():
    session_keys = {
        "json_data": None,
        "raw_json": "",
        "editor_key": 0,
        "file_processed": False,
        "show_paths": False,
        "path_results": {},
        "min_time": 0.0,
        "max_time": 100.0,
        "graph_layout": "分层布局",
        "self_implemented_nodes": set(),
        "node_connectivity": {},
        "show_real_names": False,
        "show_mapping_table": False
    }
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = session_keys[key]


def process_capability_data(raw_data):
    if not isinstance(raw_data, list):
        raise ValueError("输入数据必须是JSON数组")

    nodes = set()
    edges = []
    timestamps = []
    capability_provider_pairs = defaultdict(set)

    for event in raw_data:
        if not isinstance(event, dict):
            continue

        msg = event.get("msg", {})
        source = msg.get("source", {})
        target = msg.get("target", {})

        stamp = msg.get("header", {}).get("stamp", {})
        event_time = stamp.get("secs", 0) + stamp.get("nsecs", 0) * 1e-9
        timestamps.append(event_time)

        if source.get("capability") and source.get("provider"):
            capability_provider_pairs[source["capability"]].add(source["provider"])

        if source.get("capability"):
            real_name = f"capability:{source['capability']}"
            mapped_id = node_mapper.map_node(real_name)
            nodes.add((mapped_id, real_name, "capability"))
        if source.get("provider"):
            real_name = f"provider:{source['provider']}"
            mapped_id = node_mapper.map_node(real_name)
            nodes.add((mapped_id, real_name, "provider"))

        if target.get("capability"):
            real_name = f"capability:{target['capability']}"
            mapped_id = node_mapper.map_node(real_name)
            nodes.add((mapped_id, real_name, "capability"))
        if target.get("provider"):
            real_name = f"provider:{target['provider']}"
            mapped_id = node_mapper.map_node(real_name)
            nodes.add((mapped_id, real_name, "provider"))

    self_implemented = set()
    for cap, providers in capability_provider_pairs.items():
        if cap in providers:
            real_name = f"capability:{cap}"
            mapped_id = node_mapper.map_node(real_name)
            self_implemented.add(mapped_id)
            real_name = f"provider:{cap}"
            mapped_id = node_mapper.map_node(real_name)
            self_implemented.add(mapped_id)
    st.session_state.self_implemented_nodes = self_implemented

    min_timestamp = min(timestamps) if timestamps else 0

    node_list = []
    node_colors = {
        "capability": "#4B8BBE",
        "provider": "#FFA500",
        "unknown": "#95a5a6"
    }
    for mapped_id, real_name, node_type in nodes:
        node_color = "#9b59b6" if mapped_id in self_implemented else node_colors.get(node_type, "#95a5a6")
        node_list.append({
            "id": mapped_id,
            "real_name": real_name,
            "label": mapped_id,
            "type": node_type,
            "color": node_color,
            "size": 1500 if node_type == "capability" else 1200,
            "shape": "d" if mapped_id in self_implemented else "o"
        })

    for event in raw_data:
        if not isinstance(event, dict):
            continue

        msg = event.get("msg", {})
        source = msg.get("source", {})
        target = msg.get("target", {})

        stamp = msg.get("header", {}).get("stamp", {})
        event_time = stamp.get("secs", 0) + stamp.get("nsecs", 0) * 1e-9
        rel_time = event_time - min_timestamp

        source_id = None
        if source.get("capability"):
            real_name = f"capability:{source['capability']}"
            source_id = node_mapper.map_node(real_name)
        elif source.get("provider"):
            real_name = f"provider:{source['provider']}"
            source_id = node_mapper.map_node(real_name)

        target_id = None
        if target.get("capability"):
            real_name = f"capability:{target['capability']}"
            target_id = node_mapper.map_node(real_name)
        elif target.get("provider"):
            real_name = f"provider:{target['provider']}"
            target_id = node_mapper.map_node(real_name)

        if source_id and target_id:
            edges.append({
                "source": source_id,
                "target": target_id,
                "label": msg.get("text", "")[:50] + "..." if len(msg.get("text", "")) > 50 else msg.get("text", ""),
                "time": round(rel_time, 3),
                "color": "#2ecc71" if rel_time < 5 else "#e74c3c",
                "width": 1.0 + rel_time / 5,
                "weight": rel_time
            })

    max_time = max([e["time"] for e in edges]) if edges else 100.0

    return {
        "nodes": node_list,
        "edges": edges,
        "min_timestamp": min_timestamp,
        "max_time": max_time
    }


def json_editor_component():
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
                processed = process_capability_data(parsed)
                st.session_state.json_data = processed
                st.session_state.raw_json = json.dumps(
                    parsed,
                    indent=2,
                    ensure_ascii=False
                )
                st.session_state.max_time = processed["max_time"]
                st.session_state.editor_key += 1
                st.rerun()
            except Exception as e:
                st.error(f"保存失败: {str(e)}")

    with col2:
        st.markdown("​**​实时预览​**​")
        if st.session_state.raw_json:
            html_code = highlight(
                st.session_state.raw_json,
                JsonLexer(),
                formatter
            )
            html(f'<div class="highlight">{html_code}</div>', height=600, scrolling=True)


def build_network_graph(data):
    G = nx.DiGraph()

    for node in data["nodes"]:
        G.add_node(node["id"],
                   label=node["label"],
                   real_name=node["real_name"],
                   node_type=node["type"],
                   color=node["color"],
                   size=node["size"],
                   shape=node.get("shape", "o"),
                   subset=int(node["id"].split(":")[0] == "provider"))

    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            label=edge["label"],
            time=edge["time"],
            weight=edge["time"],
            color=edge["color"],
            width=edge["width"]
        )
    return G


def generate_network_graph(data):
    try:
        G = build_network_graph(data)

        if st.session_state.graph_layout == "分层布局":
            pos = nx.multipartite_layout(G, subset_key="subset", align="horizontal")
        else:
            pos = nx.spring_layout(G, k=1.5, iterations=100)

        fig, ax = plt.subplots(figsize=(14, 10), dpi=120)
        title = "Capability Event Flow" + (" (显示真实名称)" if st.session_state.show_real_names else "")
        plt.title(title, pad=20, fontsize=16)

        regular_nodes = [n for n in G.nodes if n not in st.session_state.self_implemented_nodes]
        self_implemented_nodes = [n for n in G.nodes if n in st.session_state.self_implemented_nodes]

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=regular_nodes,
            node_size=[G.nodes[n]['size'] for n in regular_nodes],
            node_color=[G.nodes[n]['color'] for n in regular_nodes],
            edgecolors="white",
            linewidths=1.5,
            node_shape="o",
            ax=ax
        )

        if self_implemented_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=self_implemented_nodes,
                node_size=[G.nodes[n]['size'] * 1.2 for n in self_implemented_nodes],
                node_color=[G.nodes[n]['color'] for n in self_implemented_nodes],
                edgecolors="white",
                linewidths=2,
                node_shape="d",
                ax=ax
            )

        for u, v, data in G.edges(data=True):
            arrow = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                arrowstyle="->",
                color=data['color'],
                mutation_scale=20,
                linewidth=data['width'] * 0.8,
                connectionstyle="arc3,rad=0.2",
                zorder=3
            )
            ax.add_patch(arrow)

        labels = {}
        for n in G.nodes:
            if st.session_state.show_real_names:
                real_name = G.nodes[n]['real_name']
                node_type = real_name.split(':')[0]
                labels[n] = f"{node_mapper.map_node(real_name)}\n({real_name.split(':')[1]})"
            else:
                labels[n] = G.nodes[n]['label']

        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=10,
            font_family='sans-serif',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5),
            ax=ax
        )

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Capability',
                       markerfacecolor='#4B8BBE', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Provider',
                       markerfacecolor='#FFA500', markersize=10),
            plt.Line2D([0], [0], marker='d', color='w', label='Self-Implemented',
                       markerfacecolor='#9b59b6', markersize=10),
            plt.Line2D([0], [0], color='#2ecc71', lw=2, label='Early Event (t<5s)'),
            plt.Line2D([0], [0], color='#e74c3c', lw=2, label='Late Event (t≥5s)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.axis("off")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"图表生成失败: {str(e)}")
        return None


def plot_path(G, path, title):
    path_edges = list(zip(path[:-1], path[1:]))

    plt.figure(figsize=(8, 4))

    if st.session_state.graph_layout == "分层布局":
        pos = nx.multipartite_layout(G, subset_key="subset", align="horizontal")
    else:
        pos = nx.spring_layout(G, k=0.8, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgray')

    path_colors = []
    for n in path:
        if n in st.session_state.self_implemented_nodes:
            path_colors.append('#9b59b6')
        elif 'capability' in node_mapper.get_node_type(n):
            path_colors.append('#4B8BBE')
        else:
            path_colors.append('#FFA500')

    nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=500, node_color=path_colors)

    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, arrows=True)

    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           edge_color='red', width=2, arrows=True)

    labels = {}
    for n in path:
        if st.session_state.show_real_names:
            real_name = G.nodes[n]['real_name']
            labels[n] = f"{n}\n({real_name.split(':')[1]})"
        else:
            labels[n] = n

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)


def get_node_connectivity_report(G, start_node, end_node, time_range):
    node_labels = {n: G.nodes[n]['real_name'].split(':')[1] for n in G.nodes}

    filtered_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if time_range[0] <= d['time'] <= time_range[1]
    ]
    subgraph = G.edge_subgraph(filtered_edges)

    start_reachable = list(nx.descendants(subgraph, start_node))
    end_reachable = list(nx.ancestors(subgraph, end_node))

    report = [
        "## ⛔ 路径分析结果",
        "",
        f"​**​在 {time_range[0]}-{time_range[1]}s 内​**​:",
        f"- 起始节点 `{start_node}` 无法到达目标节点 `{end_node}`",
        "",
        "### 🔍 连接性分析",
        f"从 `{start_node}` 可到达的节点:",
    ]

    for n in start_reachable[:5]:
        report.append(f"- `{n}` ({node_labels[n]})")
    if len(start_reachable) > 5:
        report.append(f"- ...共 {len(start_reachable)} 个可达节点")

    report.extend([
        "",
        f"能到达 `{end_node}` 的节点:"
    ])

    for n in end_reachable[:5]:
        report.append(f"- `{n}` ({node_labels[n]})")
    if len(end_reachable) > 5:
        report.append(f"- ...共 {len(end_reachable)} 个节点")

    report.extend([
        "",
        "### 💡 建议方案",
        "1. ​**​调整时间范围​**​ - 尝试扩大时间范围",
        "2. ​**​选择替代节点​**​ - 从以下推荐组合中选择:",
    ])

    connected_pairs = []
    all_nodes = list(G.nodes)
    for _ in range(min(50, len(all_nodes))):
        u = random.choice(all_nodes)
        reachable = nx.descendants(subgraph, u)
        if reachable:
            v = random.choice(list(reachable))
            connected_pairs.append((u, v))
            if len(connected_pairs) >= 3:
                break

    if connected_pairs:
        for u, v in connected_pairs:
            report.append(f"- `{u}` → `{v}`")
    else:
        report.append("- 当前时间范围内无可用连接")

    return "\n".join(report)


def path_analysis_panel():
    st.header("⏱️ 路径分析")

    st.checkbox("显示真实名称", key="show_real_names",
                help="切换显示映射ID或原始名称")

    if not st.session_state.json_data:
        st.warning("请先加载数据")
        return

    G = build_network_graph(st.session_state.json_data)

    all_nodes = [n["id"] for n in st.session_state.json_data["nodes"]]
    node_labels = {n["id"]: n["real_name"] for n in st.session_state.json_data["nodes"]}

    st.session_state.graph_layout = st.radio(
        "选择布局方式",
        ["分层布局", "力导向布局"],
        horizontal=True
    )

    st.subheader("时间范围过滤")
    min_time, max_time = st.slider(
        "选择分析的时间范围(秒)",
        min_value=0.0,
        max_value=st.session_state.max_time,
        value=(0.0, st.session_state.max_time),
        step=0.1,
        format="%.1f"
    )

    st.subheader("节点选择")
    col1, col2 = st.columns(2)
    with col1:
        start_node = st.selectbox(
            "起始节点",
            options=all_nodes,
            index=0,
            format_func=lambda x: f"{x} ({node_mapper.get_node_type(x)})",
            key="path_start"
        )
    with col2:
        end_node = st.selectbox(
            "目标节点",
            options=all_nodes,
            index=min(1, len(all_nodes) - 1),
            format_func=lambda x: f"{x} ({node_mapper.get_node_type(x)})",
            key="path_end"
        )

    if st.button("🔍 计算最优路径", type="primary"):
        st.session_state.show_paths = True
        try:
            filtered_edges = [
                (u, v) for u, v, d in G.edges(data=True)
                if min_time <= d['time'] <= max_time
            ]
            subgraph = G.edge_subgraph(filtered_edges)

            if not nx.has_path(subgraph, start_node, end_node):
                error_report = get_node_connectivity_report(
                    G, start_node, end_node, (min_time, max_time)
                )
                st.markdown(f'<div class="path-error">{error_report}</div>', unsafe_allow_html=True)
                return

            time_path = nx.shortest_path(subgraph, start_node, end_node, weight='time')
            time_cost = sum(G.edges[u, v]['time'] for u, v in zip(time_path[:-1], time_path[1:]))

            hop_path = nx.shortest_path(subgraph, start_node, end_node)
            hop_cost = len(hop_path) - 1

            for u, v in subgraph.edges():
                subgraph.edges[u, v]['mixed_weight'] = subgraph.edges[u, v]['time'] + 0.3
            mixed_path = nx.shortest_path(subgraph, start_node, end_node, weight='mixed_weight')
            mixed_cost = sum(G.edges[u, v]['time'] for u, v in zip(mixed_path[:-1], mixed_path[1:]))

            st.session_state.path_results = {
                "time_path": (time_path, time_cost),
                "hop_path": (hop_path, hop_cost),
                "mixed_path": (mixed_path, mixed_cost)
            }

        except Exception as e:
            st.error(f"路径计算错误: {str(e)}")

    if st.session_state.show_paths and 'path_results' in st.session_state:
        st.subheader("📊 路径分析结果")

        tab1, tab2, tab3 = st.tabs(["⏱️ 最短时间路径", "↔️ 最少跳数路径", "⚖️ 平衡路径"])

        with tab1:
            path, cost = st.session_state.path_results["time_path"]
            display_path = " → ".join([
                f"{node}\n({node_labels[node].split(':')[1]})"
                if st.session_state.show_real_names
                else node
                for node in path
            ])
            st.success(f"""
                ​**​最短时间路径​**​ (总耗时: {cost:.2f}秒)
                ```
                {display_path}
                ```
            """)
            plot_path(G, path, "时间最优路径")

        with tab2:
            path, cost = st.session_state.path_results["hop_path"]
            display_path = " → ".join([
                f"{node}\n({node_labels[node].split(':')[1]})"
                if st.session_state.show_real_names
                else node
                for node in path
            ])
            st.info(f"""
                ​**​最少跳数路径​**​ (共 {cost} 跳)
                ```
                {display_path}
                ```
            """)
            plot_path(G, path, "跳数最优路径")

        with tab3:
            path, cost = st.session_state.path_results["mixed_path"]
            display_path = " → ".join([
                f"{node}\n({node_labels[node].split(':')[1]})"
                if st.session_state.show_real_names
                else node
                for node in path
            ])
            st.warning(f"""
                ​**​平衡路径​**​ (耗时: {cost:.2f}秒)
                ```
                {display_path}
                ```
            """)
            plot_path(G, path, "平衡权重路径")


def show_mapping_table():
    if os.path.exists("node_mapping.json"):
        with open("node_mapping.json") as f:
            mapping_data = json.load(f)

        st.sidebar.markdown("### 节点映射表")
        st.sidebar.markdown('<div class="mapping-table">', unsafe_allow_html=True)

        cols = st.sidebar.columns(2)
        with cols[0]:
            st.write("​**​映射ID​**​")
            for k in sorted(mapping_data["mapped_to_real"].keys()):
                st.code(k)

        with cols[1]:
            st.write("​**​原始名称​**​")
            for k in sorted(mapping_data["mapped_to_real"].keys()):
                real_name = mapping_data["mapped_to_real"][k]
                st.code(real_name.split(':')[1] if ':' in real_name else real_name)

        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    else:
        st.sidebar.write("暂无映射数据")


def main():
    init_session()

    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🤖 机器人能力事件流分析系统</h1>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("上传JSON文件", type=["json"], key="uploaded_file")

    if uploaded_file and not st.session_state.file_processed:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            raw_data = json.loads(content)
            processed_data = process_capability_data(raw_data)

            st.session_state.json_data = processed_data
            st.session_state.raw_json = json.dumps(
                raw_data,
                indent=2,
                ensure_ascii=False
            )
            st.session_state.max_time = processed_data["max_time"]
            st.session_state.file_processed = True
            st.success("文件处理成功！")
        except Exception as e:
            st.error(f"文件处理失败: {str(e)}")

    st.sidebar.checkbox("显示映射关系表", key="show_mapping_table")
    if st.session_state.show_mapping_table:
        show_mapping_table()

    col1, col2, col3 = st.columns([2.5, 4, 1.5])

    with col1:
        st.header("📂 数据输入")
        if st.session_state.raw_json:
            json_editor_component()
        else:
            st.info("请上传JSON文件或直接编辑右侧内容")

    with col2:
        st.header("🌐 网络可视化")
        if st.session_state.json_data:
            fig = generate_network_graph(st.session_state.json_data)
            if fig:
                st.pyplot(fig)
        else:
            st.info("等待数据输入...")

    with col3:
        if st.session_state.json_data:
            path_analysis_panel()
        else:
            st.info("请先上传数据")


if __name__ == "__main__":
    main()