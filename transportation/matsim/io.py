import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import json
import os
from .network import TransportNetwork, TransportNode, TransportLink

def read_network_from_csv(nodes_file: str, links_file: str, weights_file: Optional[str] = None) -> TransportNetwork:
    """
    Read a transport network from CSV files
    
    Args:
        nodes_file: Path to CSV file with node data
        links_file: Path to CSV file with link data
        weights_file: Optional path to CSV file with weight data
    
    Returns:
        TransportNetwork object
    """
    # Read nodes
    nodes_df = pd.read_csv(nodes_file)
    
    # Read links
    links_df = pd.read_csv(links_file)
    
    # Create network
    network = TransportNetwork()
    
    # Add nodes
    for _, row in nodes_df.iterrows():
        node = TransportNode(
            node_id=str(row['node_id']),
            x=float(row['x']),
            y=float(row['y']),
            name=str(row.get('name', row['node_id'])),
            attributes={k: row[k] for k in row.index if k not in ['node_id', 'x', 'y', 'name']}
        )
        network.add_node(node)
    
    # Add links
    for _, row in links_df.iterrows():
        link = TransportLink(
            link_id=str(row['link_id']),
            from_node=str(row['from_node']),
            to_node=str(row['to_node']),
            length=float(row['length']),
            capacity=float(row['capacity']),
            free_speed=float(row['free_speed']),
            attributes={k: row[k] for k in row.index if k not in ['link_id', 'from_node', 'to_node', 'length', 'capacity', 'free_speed']}
        )
        network.add_link(link)
    
    # Add weights if provided
    if weights_file and os.path.exists(weights_file):
        weights_df = pd.read_csv(weights_file)
        for _, row in weights_df.iterrows():
            network.add_weight_data(
                link_id=str(row['link_id']),
                time_period=str(row['time_period']),
                weight=float(row['weight'])
            )
    
    return network

def write_network_to_csv(network: TransportNetwork, output_dir: str):
    """
    Write a transport network to CSV files
    
    Args:
        network: TransportNetwork object
        output_dir: Directory to write CSV files to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Write nodes
    nodes_data = []
    for node_id, node in network.nodes.items():
        node_data = {
            'node_id': node.node_id,
            'x': node.x,
            'y': node.y,
            'name': node.name
        }
        node_data.update(node.attributes)
        nodes_data.append(node_data)
    
    nodes_df = pd.DataFrame(nodes_data)
    nodes_df.to_csv(os.path.join(output_dir, 'nodes.csv'), index=False)
    
    # Write links
    links_data = []
    for link_id, link in network.links.items():
        link_data = {
            'link_id': link.link_id,
            'from_node': link.from_node,
            'to_node': link.to_node,
            'length': link.length,
            'capacity': link.capacity,
            'free_speed': link.free_speed
        }
        link_data.update(link.attributes)
        links_data.append(link_data)
    
    links_df = pd.DataFrame(links_data)
    links_df.to_csv(os.path.join(output_dir, 'links.csv'), index=False)
    
    # Write weights
    weights_data = []
    for link_id, link in network.links.items():
        for time_period, weight in link.weights_history.items():
            weights_data.append({
                'link_id': link_id,
                'time_period': time_period,
                'weight': weight
            })
    
    weights_df = pd.DataFrame(weights_data)
    if not weights_df.empty:
        weights_df.to_csv(os.path.join(output_dir, 'weights.csv'), index=False)

def read_network_from_matsim(network_file: str, weights_file: Optional[str] = None) -> TransportNetwork:
    """
    Read a transport network from MATSim XML format
    
    Args:
        network_file: Path to MATSim network XML file
        weights_file: Optional path to CSV file with weight data
    
    Returns:
        TransportNetwork object
    """
    tree = ET.parse(network_file)
    root = tree.getroot()
    
    network = TransportNetwork()
    
    # Read nodes
    for node_elem in root.findall(".//node"):
        node_id = node_elem.get('id')
        x = float(node_elem.get('x'))
        y = float(node_elem.get('y'))
        
        attributes = {}
        for attr in node_elem.findall("./attribute"):
            attributes[attr.get('name')] = attr.text
        
        node = TransportNode(
            node_id=node_id,
            x=x,
            y=y,
            name=attributes.get('name', node_id),
            attributes=attributes
        )
        network.add_node(node)
    
    # Read links
    for link_elem in root.findall(".//link"):
        link_id = link_elem.get('id')
        from_node = link_elem.get('from')
        to_node = link_elem.get('to')
        length = float(link_elem.get('length'))
        capacity = float(link_elem.get('capacity'))
        free_speed = float(link_elem.get('freespeed'))
        
        attributes = {}
        for attr in link_elem.findall("./attribute"):
            attributes[attr.get('name')] = attr.text
        
        link = TransportLink(
            link_id=link_id,
            from_node=from_node,
            to_node=to_node,
            length=length,
            capacity=capacity,
            free_speed=free_speed,
            attributes=attributes
        )
        network.add_link(link)
    
    # Add weights if provided
    if weights_file and os.path.exists(weights_file):
        weights_df = pd.read_csv(weights_file)
        for _, row in weights_df.iterrows():
            network.add_weight_data(
                link_id=str(row['link_id']),
                time_period=str(row['time_period']),
                weight=float(row['weight'])
            )
    
    return network

def write_network_to_matsim(network: TransportNetwork, output_file: str):
    """
    Write a transport network to MATSim XML format
    
    Args:
        network: TransportNetwork object
        output_file: Path to output XML file
    """
    root = ET.Element("network")
    nodes = ET.SubElement(root, "nodes")
    
    # Write nodes
    for node_id, node in network.nodes.items():
        node_elem = ET.SubElement(nodes, "node")
        node_elem.set("id", node.node_id)
        node_elem.set("x", str(node.x))
        node_elem.set("y", str(node.y))
        
        # Add attributes
        for key, value in node.attributes.items():
            attr = ET.SubElement(node_elem, "attribute")
            attr.set("name", key)
            attr.text = str(value)
    
    links = ET.SubElement(root, "links")
    
    # Write links
    for link_id, link in network.links.items():
        link_elem = ET.SubElement(links, "link")
        link_elem.set("id", link.link_id)
        link_elem.set("from", link.from_node)
        link_elem.set("to", link.to_node)
        link_elem.set("length", str(link.length))
        link_elem.set("capacity", str(link.capacity))
        link_elem.set("freespeed", str(link.free_speed))
        
        # Add attributes
        for key, value in link.attributes.items():
            attr = ET.SubElement(link_elem, "attribute")
            attr.set("name", key)
            attr.text = str(value)
    
    # Create XML tree and write to file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True) 