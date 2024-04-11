from .graph_manager import GraphManager

def main():
    graph_manager = GraphManager()
    graph_manager.recive()
    graph_manager.update()
    graph_manager.send()

if __name__ == '__main__':
    main()