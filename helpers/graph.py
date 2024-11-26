from IPython.display import Image, display

def draw_graph(graph):
    """
    Draw the graph with the given Graph object.

    The graph is drawn using the Mermaid library and displayed using IPython.display.Image.

    Parameters
    ----------
    graph : langchain.graph.Graph
        The Graph object to draw.
    """
    display(Image(graph.get_graph().draw_mermaid_png()))