import matsim
import subprocess
import os
import random
import numpy as np
import json
import requests
from typing import List, Dict, Any
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

# Add function to the matsim module
def write_population(population, output_file):
    """Write a MATSim population to an XML file
    
    Args:
        population: MATSim population object
        output_file: Path to output XML file
    """
    # Create the root element
    root = ET.Element("population")
    
    # Add DOCTYPE 
    root.append(ET.Comment('DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd"'))
    
    # Add each person
    for person_id, person in population.persons.items():
        person_elem = ET.SubElement(root, "person", id=str(person_id))
        
        # Add plans
        for i, plan in enumerate(person.plans):
            selected = "yes" if i == person.selected_plan_index else "no"
            plan_elem = ET.SubElement(person_elem, "plan", selected=selected)
            
            # Add activities and legs alternating
            elements = []
            
            # Add activities
            for i, activity in enumerate(plan.activities):
                # Get basic attributes, accommodating different possible property names
                act_attrs = {"type": activity.type}
                
                # Handle link attribute - could be .link or .link_id
                if hasattr(activity, "link"):
                    act_attrs["link"] = activity.link
                elif hasattr(activity, "link_id"):
                    act_attrs["link"] = activity.link_id
                    
                # Handle coordinates
                if hasattr(activity, "x") and hasattr(activity, "y"):
                    act_attrs["x"] = str(activity.x)
                    act_attrs["y"] = str(activity.y)
                
                # Handle end time
                if hasattr(activity, "end_time") and activity.end_time is not None:
                    act_attrs["end_time"] = format_time(activity.end_time)
                
                act_elem = ET.SubElement(plan_elem, "activity", act_attrs)
                
                # Add leg after each activity except the last one
                if i < len(plan.activities) - 1 and i < len(plan.legs):
                    leg = plan.legs[i]
                    # Get mode - could be .mode or .transport_mode
                    leg_mode = leg.mode if hasattr(leg, "mode") else getattr(leg, "transport_mode", "car")
                    leg_elem = ET.SubElement(plan_elem, "leg", mode=leg_mode)
                    
                    # Add route if available
                    if hasattr(leg, "route") and leg.route is not None:
                        route = leg.route
                        route_type = getattr(route, "type", "links")
                        
                        # Handle different ways route links might be stored
                        link_ids = None
                        if hasattr(route, "link_ids"):
                            link_ids = route.link_ids
                        elif hasattr(route, "links"):
                            link_ids = route.links
                            
                        if link_ids is not None:
                            route_elem = ET.SubElement(leg_elem, "route", type=route_type)
                            # Handle if links is a string or a list
                            if isinstance(link_ids, str):
                                route_elem.text = link_ids
                            else:
                                route_elem.text = " ".join(link_ids)
    
    # Write to file with pretty formatting
    tree = ET.ElementTree(root)
    xml_str = ET.tostring(root, encoding='utf-8')
    
    # Pretty print XML
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="    ")
    
    # Remove extra blank lines and write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">\n')
        for line in pretty_xml.split("\n")[1:]:  # Skip the XML declaration line
            if line.strip():  # Skip empty lines
                f.write(line + "\n")
                
def format_time(seconds_or_time_str):
    """Format time as HH:MM:SS
    
    Args:
        seconds_or_time_str: Either seconds (int/float) or time string
    
    Returns:
        Formatted time string
    """
    if isinstance(seconds_or_time_str, (int, float)):
        # Convert seconds to HH:MM:SS
        hours, remainder = divmod(seconds_or_time_str, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    else:
        # Already a string, return as is
        return seconds_or_time_str

class MATSimEnvironment:
    """Environment for MATSim simulation with agent feedback"""
    
    def __init__(self, network_file: str, plans_file: str, config_file: str, output_dir: str):
        """Initialize the MATSim environment
        
        Args:
            network_file: Path to the network XML file
            plans_file: Path to the plans XML file
            config_file: Path to the config XML file
            output_dir: Directory to store output files
        """
        self.network_file = network_file
        self.plans_file = plans_file
        self.config_file = config_file
        self.output_dir = output_dir
        self.iteration = 0
        self.network = None
        self.population = None
        
        # Load network and population
        self.load_files()
    
    def load_files(self):
        """Load network and population files"""
        self.network = matsim.read_network(self.network_file)
        print(f"Loaded network with {len(self.network.nodes)} nodes and {len(self.network.links)} links")
    
    def run_iteration(self, matsim_jar_path: str) -> Dict[str, Any]:
        """Run a single MATSim iteration
        
        Args:
            matsim_jar_path: Path to the MATSim JAR file
            
        Returns:
            Dictionary containing network state information
        """
        # Create iteration-specific plans file
        iter_plans_file = os.path.join(self.output_dir, f"plans_iter_{self.iteration}.xml")
        
        # Update config to use the new plans file
        # (In a real implementation, you'd use matsim to modify the config)
        
        # Run MATSim (one iteration)
        cmd = [
            "java", "-Xmx4g", 
            "-cp", matsim_jar_path,
            "org.matsim.run.Controler", 
            self.config_file,
            "--config:controler.lastIteration", "0"  # Run only one iteration
        ]
        
        print(f"Running MATSim iteration {self.iteration}...")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"MATSim execution failed: {process.stderr}")
            return {}
        
        # Extract network state from MATSim output
        network_state = self._parse_network_state()
        
        self.iteration += 1
        return network_state
    
    def _parse_network_state(self) -> Dict[str, Any]:
        """Parse MATSim output to extract network state
        
        In a real implementation, you would:
        1. Parse the events file to get link travel times
        2. Parse the experienced plans file
        3. Compute statistics about network congestion
        
        Returns:
            Dictionary with link IDs as keys and travel stats as values
        """
        # Placeholder - in real implementation, parse MATSim output files
        # For example, you could parse events.xml to get travel times
        
        # Simulating link statistics for demonstration purposes
        link_stats = {}
        for link_id in self.network.links:
            link = self.network.links[link_id]
            # Simulate travel times - replace with actual parsing
            free_flow_time = link.length / link.freespeed * 60  # minutes
            # Simulate congestion based on capacity
            congestion_factor = random.uniform(1.0, 3.0) * (2000 / (link.capacity + 500)) 
            travel_time = free_flow_time * congestion_factor
            
            link_stats[link_id] = {
                'travel_time': travel_time,  # minutes
                'volume': random.randint(0, int(link.capacity * 0.8)),  # vehicles
                'congestion': min(1.0, (congestion_factor - 1) / 2),  # congestion index (0-1)
                'length': link.length,  # meters
                'capacity': link.capacity,  # vehicles per hour
                'freespeed': link.freespeed * 3.6,  # km/h
            }
        
        return link_stats

# this llm agent is just for testing, later, we will get to 
class LLMAgent:
    """Agent that uses GPT-4 to make decisions about plan updates"""
    
    def __init__(self, person_id: str, environment: MATSimEnvironment, openai_api_key: str):
        """Initialize agent
        
        Args:
            person_id: ID of the person in the MATSim population
            environment: MATSimEnvironment instance
            openai_api_key: API key for OpenAI
        """
        self.person_id = person_id
        self.environment = environment
        self.person = environment.population.persons[person_id]
        self.api_key = openai_api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def update_plan(self, network_state: Dict[str, Any]):
        """Update the agent's plan based on network state using GPT-4
        
        Args:
            network_state: Dictionary with network state information
        """
        if not self.person.plans:
            print(f"Warning: Person {self.person_id} has no plans")
            return
        
        # Get the selected plan
        selected_plan = self.person.selected_plan
        
        # Extract current plan data
        plan_data = self._extract_plan_data(selected_plan)
        
        # Get relevant network data for the routes used by this agent
        network_data = self._extract_relevant_network_data(network_state, selected_plan)
        
        # Generate prompt for GPT-4
        prompt = self._generate_prompt(plan_data, network_data)
        
        # Get GPT-4 response
        gpt_response = self._query_gpt4(prompt)
        
        # Parse and apply the updated plan
        self._apply_gpt_suggestions(gpt_response, selected_plan)
        
        print(f"Agent {self.person_id} has updated their plan using GPT-4")
    
    def _extract_plan_data(self, plan):
        """Extract relevant data from a plan for the LLM
        
        Args:
            plan: MATSim plan
            
        Returns:
            Dictionary with plan data
        """
        activities = []
        legs = []
        
        for activity in plan.activities:
            act_data = {
                "type": activity.type,
                "location": {
                    "x": activity.x,
                    "y": activity.y,
                    "link": activity.link
                },
                "start_time": activity.start_time if hasattr(activity, 'start_time') else None,
                "end_time": activity.end_time if hasattr(activity, 'end_time') else None,
            }
            activities.append(act_data)
        
        for leg in plan.legs:
            route_links = leg.route.link_ids if hasattr(leg, 'route') else []
            leg_data = {
                "mode": leg.mode,
                "dep_time": leg.dep_time if hasattr(leg, 'dep_time') else None,
                "trav_time": leg.trav_time if hasattr(leg, 'trav_time') else None,
                "route_links": route_links
            }
            legs.append(leg_data)
        
        return {
            "activities": activities,
            "legs": legs
        }
    
    def _extract_relevant_network_data(self, network_state, plan):
        """Extract network data relevant to this agent's routes
        
        Args:
            network_state: Full network state
            plan: MATSim plan
            
        Returns:
            Dictionary with relevant network data
        """
        relevant_links = set()
        
        # Collect all links used in the plan
        for leg in plan.legs:
            if hasattr(leg, 'route') and hasattr(leg.route, 'link_ids'):
                for link_id in leg.route.link_ids:
                    relevant_links.add(link_id)
        
        # Get data for these links
        relevant_data = {link_id: network_state[link_id] for link_id in relevant_links 
                         if link_id in network_state}
        
        return relevant_data
    
    def _generate_prompt(self, plan_data, network_data):
        """Generate a prompt for GPT-4
        
        Args:
            plan_data: Agent's current plan
            network_data: Relevant network state
            
        Returns:
            Prompt string
        """
        prompt = f"""You are a transportation planning AI assistant for agent {self.person_id}.
        
Your task is to analyze the current transportation network conditions and suggest modifications to the agent's daily schedule to minimize travel time and avoid congestion.

CURRENT PLAN:
{json.dumps(plan_data, indent=2)}

NETWORK CONDITIONS:
{json.dumps(network_data, indent=2)}

Based on the network conditions:
1. Identify congested links in the agent's route (congestion > 0.5)
2. Suggest adjustments to departure times (earlier or later) to avoid peak congestion
3. Optionally suggest alternative routes if you can see better options using the given network

Please provide your recommendations in the following JSON format:
```json
{
  "activity_adjustments": [
    {"activity_index": 0, "new_end_time": "HH:MM:SS"},
    ...
  ],
  "routing_suggestions": [
    {"leg_index": 0, "suggested_links": ["link_id1", "link_id2", ...]}
    ...
  ],
  "explanation": "Your explanation of the changes"
}
```

Only include activities or legs that need changes. Be sure the JSON is properly formatted.
"""
        return prompt
    
    def _query_gpt4(self, prompt):
        """Query GPT-4 API with the prompt
        
        Args:
            prompt: Prompt string
            
        Returns:
            Response from GPT-4
        """
        try:
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a transportation planning assistant that helps optimize travel plans."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        
        except Exception as e:
            print(f"Error querying GPT-4: {e}")
            # Return a default empty response if API call fails
            return """```json
            {
              "activity_adjustments": [],
              "routing_suggestions": [],
              "explanation": "Failed to generate recommendations due to API error."
            }
            ```"""
    
    def _apply_gpt_suggestions(self, gpt_response, plan):
        """Parse GPT-4 response and apply changes to the plan
        
        Args:
            gpt_response: String response from GPT-4
            plan: MATSim plan to modify
        """
        try:
            # Extract JSON from the response
            json_start = gpt_response.find("```json") + 7 if "```json" in gpt_response else gpt_response.find("{")
            json_end = gpt_response.find("```", json_start) if "```" in gpt_response[json_start:] else len(gpt_response)
            json_str = gpt_response[json_start:json_end].strip()
            
            # Parse the JSON
            recommendations = json.loads(json_str)
            
            # Apply activity time adjustments
            for adjustment in recommendations.get("activity_adjustments", []):
                activity_index = adjustment["activity_index"]
                new_end_time = adjustment["new_end_time"]
                
                if 0 <= activity_index < len(plan.activities):
                    plan.activities[activity_index].end_time = new_end_time
                    print(f"  Adjusted activity {activity_index} end time to {new_end_time}")
            
            # Apply route suggestions (in a real implementation)
            # This is more complex because you'd need to create new Route objects
            for suggestion in recommendations.get("routing_suggestions", []):
                leg_index = suggestion["leg_index"]
                suggested_links = suggestion["suggested_links"]
                
                if 0 <= leg_index < len(plan.legs):
                    # In a real implementation, you'd need to create a proper Route object
                    if hasattr(plan.legs[leg_index], 'route'):
                        plan.legs[leg_index].route.link_ids = suggested_links
                        print(f"  Updated route for leg {leg_index}")
            
            print(f"  Explanation: {recommendations.get('explanation', 'No explanation provided')}")
            
        except Exception as e:
            print(f"Error applying GPT suggestions: {e}")


def run_simulation(network_file: str, plans_file: str, config_file: str, 
                  output_dir: str, matsim_jar_path: str, openai_api_key: str, num_iterations: int):
    """Run the simulation for multiple iterations
    
    Args:
        network_file: Path to network XML file
        plans_file: Path to plans XML file
        config_file: Path to config XML file
        output_dir: Directory to store output files
        matsim_jar_path: Path to MATSim JAR file
        openai_api_key: OpenAI API key for GPT-4
        num_iterations: Number of iterations to run
    """
    # Initialize environment
    env = MATSimEnvironment(network_file, plans_file, config_file, output_dir)
    
    # Create LLM agents (one per person in population)
    agents = []
    for person_id in env.population.persons:
        agents.append(LLMAgent(person_id, env, openai_api_key))
    
    print(f"Created {len(agents)} LLM agents")
    
    # Run simulation iterations
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        
        # Run MATSim and get network state
        network_state = env.run_iteration(matsim_jar_path)
        
        # Update agent plans based on network state using GPT-4
        for agent in agents:
            agent.update_plan(network_state)
        
        # Write updated plans file for the next iteration
        write_population(env.population, 
                               os.path.join(output_dir, f"updated_plans_{i}.xml"))
    
    print(f"Simulation completed after {num_iterations} iterations")


if __name__ == "__main__":
    # Example usage (replace with actual file paths)
    network_file = "example_network.xml"
    plans_file = "example_plans.xml"
    config_file = "example_config.xml"
    output_dir = "output"
    matsim_jar_path = "path/to/matsim.jar"
    
    # Replace with your actual OpenAI API key
    openai_api_key = "your_openai_api_key_here"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation for 5 iterations
    run_simulation(network_file, plans_file, config_file, 
                  output_dir, matsim_jar_path, openai_api_key, num_iterations=5)