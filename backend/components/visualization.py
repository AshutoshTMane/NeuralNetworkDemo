from pyvis.network import Network

def visualize_interactive_model(input_size, hidden_layers, output_size):
    net = Network(height="600px", width="900px", directed=True)

    # Define colors
    input_color = "#FFB703"
    hidden_color = "#219EBC"
    output_color = "#FB8500"
    edge_color = "#8ECAE6"

    # Input node
    net.add_node("Input", label=f"Input ({input_size})", color=input_color, shape="ellipse", font={"size": 18})

    previous_layer = "Input"

    # Hidden layers
    for i, layer_size in enumerate(hidden_layers):
        layer_name = f"Hidden {i+1}"
        net.add_node(layer_name, label=f"Hidden {i+1} ({layer_size})", color=hidden_color, shape="box", font={"size": 16})
        net.add_edge(previous_layer, layer_name, color=edge_color, arrows="to")
        previous_layer = layer_name

    # Output node
    net.add_node("Output", label=f"Output ({output_size})", color=output_color, shape="ellipse", font={"size": 18})
    net.add_edge(previous_layer, "Output", color=edge_color, arrows="to")

    # Improved Layout
    net.set_options("""
    var options = {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -1000,
                "springLength": 150,
                "springConstant": 0.05
            }
        },
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "LR",
                "levelSeparation": 150,
                "nodeSpacing": 200,
                "treeSpacing": 300,
                "sortMethod": "directed"
            }
        },
        "interaction": {
            "hover": true,
            "dragNodes": true,
            "zoomView": true
        }
    }
    """)

    html_file = "interactive_model.html"
    net.save_graph(html_file)
    return html_file
