import json
import re
from pathlib import Path


def get_symfony_classes(project_path):
    """Get a list of classes in the given Symfony project"""
    composer_json = Path(project_path) / 'composer.json'
    if not composer_json.exists():
        return []

    composer_content = json.loads(composer_json.read_text())
    src_dir = Path(project_path) / 'src'
    if not src_dir.exists():
        return []

    return [Path(str(file)) for file in src_dir.rglob('*.php')]


def get_class_relationships(classes):
    """Get the relationship between classes"""
    graph = {}
    for file in classes:
        with open(file, 'r') as f:
            code = f.read()
            class_name = re.search(r'class\s+(\w+)', code).group(1)

            if class_name not in graph:
                graph[class_name] = []

            # Extract methods from the class
            methods = re.findall(r'df\s+\w+(?=\s*\(\s*.*', code)
            for method in methods:
                graph[class_name].append(method)

    return graph


def print_graph(graph):
    """Print the relationship between classes"""
    for node, edges in graph.items():
        print(f"Class {node} has methods: {', '.join(edges)}")
        for neighbor in graph:
            if neighbor != node and any(method in graph[neighbor] for method in edges):
                print(f"  - {neighbor}")


def main(project_path='/home/andres/workspace/oquanta-clients-webapp'):
    print(f"Exploring project at {project_path}")
    classes = list(get_symfony_classes(project_path))
    relationships = get_class_relationships(classes)
    print_graph(relationships)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a Symfony project directory and its contents")
    parser.add_argument("project_path", help="Path to the Symfony project",
                        default='/home/andres/workspace/oquanta-clients-webapp')
    args = parser.parse_args()
    main(args.project_path)
