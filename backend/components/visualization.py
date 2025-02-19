from pyvis.network import Network

def visualize_interactive_model(input_size, hidden_layers, output_size):
    # Create a directed network graph
    net = Network(height="600px", width="900px", directed=True)

    # Define colors for different types of nodes
    input_color = "#FFB703"    # Orange shade for input
    hidden_color = "#219EBC"   # Blue shade for hidden layers
    output_color = "#FB8500"   # Darker orange for output
    edge_color = "#8ECAE6"     # Light blue for edges

    # Add the input node with label and styling
    net.add_node("Input", label=f"Input ({input_size})", color=input_color, shape="ellipse", font={"size": 18})

    previous_layer = "Input"  # Track the previous layer to connect edges

    # Loop through hidden layers and add them to the network
    for i, layer_size in enumerate(hidden_layers):
        layer_name = f"Hidden {i+1}"  # Naming each hidden layer sequentially
        net.add_node(layer_name, label=f"Hidden {i+1} ({layer_size})", color=hidden_color, shape="box", font={"size": 16})
        
        # Create a directed edge from the previous layer to the current hidden layer
        net.add_edge(previous_layer, layer_name, color=edge_color, arrows="to")
        
        # Update previous layer to the current one for the next iteration
        previous_layer = layer_name

    # Add the output node
    net.add_node("Output", label=f"Output ({output_size})", color=output_color, shape="ellipse", font={"size": 18})
    
    # Connect the last hidden layer to the output node
    net.add_edge(previous_layer, "Output", color=edge_color, arrows="to")

    # Customize visualization settings to improve layout and interaction
    net.set_options("""
    var options = {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -1000,  # Controls attraction/repulsion between nodes
                "springLength": 150,  # Distance between connected nodes
                "springConstant": 0.05  # Strength of the connection
            }
        },
        "layout": {
            "hierarchical": {
                "enabled": true,  # Enables hierarchical layout
                "direction": "LR",  # Layout direction (Left to Right)
                "levelSeparation": 150,  # Space between layers
                "nodeSpacing": 200,  # Space between nodes in the same layer
                "treeSpacing": 300,  # Space between branches
                "sortMethod": "directed"  # Ensures logical order of nodes
            }
        },
        "interaction": {
            "hover": true,  # Highlight nodes when hovered
            "dragNodes": true,  # Allow dragging of nodes
            "zoomView": true  # Enable zooming
        }
    }
    """)

    # Save the interactive visualization as an HTML file
    html_file = "interactive_model.html"
    net.save_graph(html_file)
    return html_file  # Return the file path for viewing
