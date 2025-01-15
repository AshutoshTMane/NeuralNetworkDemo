from pyvis.network import Network


# Function to visualize the model using Pyvis
def visualize_interactive_model(input_size, hidden_layers, output_size):
    net = Network(height="500px", width="800px", directed=True)

    net.add_node("Input", label="Input ({} nodes)".format(input_size), color="#f4a261", shape="circle")
    previous_layer = "Input"

    for i, layer_size in enumerate(hidden_layers):
        layer_name = f"Hidden Layer {i+1}"
        net.add_node(layer_name, label=f"Hidden Layer {i+1} ({layer_size} nodes)", color="#2a9d8f", shape="circle")
        net.add_edge(previous_layer, layer_name)
        previous_layer = layer_name

    net.add_node("Output", label="Output ({} nodes)".format(output_size), color="#e76f51", shape="circle")
    net.add_edge(previous_layer, "Output")

    net.set_options("""
    var options = {
        "physics": {
            "enabled": true
        },
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "LR",
                "sortMethod": "directed"
            }
        },
        "interaction": {
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)

    html_file = "interactive_model.html"
    net.save_graph(html_file)
    return html_file