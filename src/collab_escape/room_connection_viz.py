import networkx as nx
import matplotlib.pyplot as plt

# Define the rooms and their connections
rooms = {
    "Control Room": ["Abandoned Warehouse", "Creepy Cabin", "Asylum"],
    "Abandoned Warehouse": ["Control Room", "Torture Chamber", "Haunted Forest"],
    "Haunted Forest": ["Abandoned Warehouse", "Asylum", "Creepy Cabin", "Cemetery"],
    "Cemetery": ["Haunted Forest", "Creepy Cabin"],
    "Creepy Cabin": ["Control Room", "Haunted Forest", "Cemetery", "Torture Chamber"],
    "Torture Chamber": ["Abandoned Warehouse", "Creepy Cabin", "Asylum"],
    "Asylum": ["Control Room", "Haunted Forest", "Torture Chamber"]
}

# Create a new graph
G = nx.Graph()

# Add edges to the graph
for room, neighbors in rooms.items():
    for neighbor in neighbors:
        G.add_edge(room, neighbor)

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # positions for all nodes, you can also use other layout algorithms
nx.draw(G, pos, with_labels=True, node_color="lightblue", font_weight="bold", node_size=700, font_size=18)
plt.title("Room Connections")

# Save the plot to a file
plt.savefig("room_connections.png")
