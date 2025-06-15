from sumo_env import SUMOEnvironment
from typing import List, Dict, Any

class LLMAgent:
    """Agent that uses LLM to make decisions about route updates, including multimodal options"""
    
    def __init__(self, vehicle_id: str, environment: SUMOEnvironment, openai_api_key: str):
        """Initialize agent
        
        Args:
            vehicle_id: ID of the vehicle in the SUMO simulation
            environment: SUMOEnvironment instance
            openai_api_key: API key for OpenAI
        """
        self.vehicle_id = vehicle_id
        self.environment = environment
        self.openai_api_key = openai_api_key
        self.route_history = []
        self.multimodal_journey = False  # Whether this agent is using multimodal transport
        self.transport_modes = []  # List of transport modes used
        
    def update_route(self, network_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update the agent's route based on network state using LLM
        
        Args:
            network_state: Dictionary with network state information
            
        Returns:
            Dictionary with route update information
        """
        print(f"Updating route for vehicle {self.vehicle_id}")
        
        # Check if vehicle exists in environment
        if self.vehicle_id not in self.environment.vehicles:
            print(f"  Vehicle {self.vehicle_id} not found in simulation")
            return {}
        
        # Extract vehicle data
        vehicle_data = self._extract_vehicle_data()
        
        # Extract relevant network data
        network_data = self._extract_relevant_network_data(network_state)
        
        # Extract public transit data
        transit_data = self._extract_public_transit_data(network_state)
        
        # Generate prompt for LLM
        prompt = self._generate_prompt(vehicle_data, network_data, transit_data)
        
        # Query LLM
        try:
            llm_response = self._query_llm(prompt)
            
            # Parse and apply LLM suggestions
            route_update = self._parse_llm_suggestions(llm_response)
            return route_update
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return {}
    
    def _extract_vehicle_data(self) -> Dict[str, Any]:
        """Extract relevant data from a vehicle
        
        Returns:
            Dictionary with vehicle data
        """
        vehicle_data = self.environment.vehicles.get(self.vehicle_id, {})
        
        # Format the data for the LLM prompt
        formatted_data: Dict[str, Any] = {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": vehicle_data.get("type", "unknown"),
            "current_route": vehicle_data.get("route_id", "unknown"),
            "edge_history": vehicle_data.get("edge_history", []),
            "travel_times": vehicle_data.get("travel_times", {}),
            "is_public_transit": vehicle_data.get("is_public_transit", False),
            "multimodal_journey": self.multimodal_journey,
            "transport_modes": self.transport_modes
        }
        
        return formatted_data
    
    def _extract_relevant_network_data(self, network_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract network data relevant to this agent's routes
        
        Args:
            network_state: Full network state
            
        Returns:
            Dictionary with relevant network data
        """
        relevant_data = {}
        
        # Skip public transit key if present
        if "public_transit" in network_state:
            network_state_edges = {k: v for k, v in network_state.items() if k != "public_transit"}
        else:
            network_state_edges = network_state
        
        # Get vehicle's edge history
        vehicle_data = self.environment.vehicles.get(self.vehicle_id, {})
        edge_history = vehicle_data.get("edge_history", [])
        
        # Include data for edges the vehicle has traveled on
        for edge_id in edge_history:
            if edge_id in network_state_edges:
                relevant_data[edge_id] = network_state_edges[edge_id]
        
        # Include data for all edges if we have limited history
        if len(relevant_data) < 5:
            # Add some of the most congested edges
            congested_edges = sorted(
                [(edge_id, data["travel_time"]) for edge_id, data in network_state_edges.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for edge_id, _ in congested_edges:
                if edge_id not in relevant_data:
                    relevant_data[edge_id] = network_state_edges[edge_id]
        
        return relevant_data
    
    def _extract_public_transit_data(self, network_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract public transit data from network state
        
        Args:
            network_state: Full network state
            
        Returns:
            Dictionary with public transit data
        """
        transit_data = {
            "nearby_stops": [],
            "relevant_buses": [],
            "relevant_trains": []
        }
        
        # Skip if public transit data is not available
        if "public_transit" not in network_state:
            return transit_data
        
        pt_data = network_state["public_transit"]
        
        # Get vehicle's current edge
        vehicle_data = self.environment.vehicles.get(self.vehicle_id, {})
        current_edge = vehicle_data.get("edge_history", [])[-1] if vehicle_data.get("edge_history") else None
        
        if not current_edge:
            return transit_data
        
        # Find nearby stops (stops on the current edge or recent history)
        edge_history = vehicle_data.get("edge_history", [])[-5:]  # Last 5 edges
        
        for stop_id, stop_data in pt_data.get("stops", {}).items():
            stop_edge = stop_data.get("edge_id")
            if stop_edge in edge_history:
                transit_data["nearby_stops"].append({
                    "id": stop_id,
                    "edge_id": stop_edge,
                    "position": stop_data.get("position"),
                    "waiting_vehicles": stop_data.get("waiting_vehicles", [])
                })
        
        # Find relevant buses (buses that pass through nearby stops)
        for bus_id, bus_data in pt_data.get("buses", {}).items():
            next_stop = bus_data.get("next_stop")
            for stop in transit_data["nearby_stops"]:
                if next_stop == stop["id"] or stop["id"] in bus_data.get("stops", []):
                    transit_data["relevant_buses"].append({
                        "id": bus_id,
                        "line": bus_data.get("line", "unknown"),
                        "current_edge": bus_data.get("current_edge"),
                        "next_stop": next_stop
                    })
                    break
        
        # Find relevant trains (trains that pass through nearby stops)
        for train_id, train_data in pt_data.get("trains", {}).items():
            next_stop = train_data.get("next_stop")
            for stop in transit_data["nearby_stops"]:
                if next_stop == stop["id"] or stop["id"] in train_data.get("stops", []):
                    transit_data["relevant_trains"].append({
                        "id": train_id,
                        "line": train_data.get("line", "unknown"),
                        "current_edge": train_data.get("current_edge"),
                        "next_stop": next_stop
                    })
                    break
        
        return transit_data
    
    def _generate_prompt(self, vehicle_data: Dict[str, Any], 
                        network_data: Dict[str, Any], 
                        transit_data: Dict[str, Any]) -> str:
        """Generate a prompt for the LLM
        
        Args:
            vehicle_data: Vehicle's current data
            network_data: Relevant network state
            transit_data: Public transit data
            
        Returns:
            Prompt string
        """
        prompt = f"""
You are an intelligent navigation system for vehicle {vehicle_data['vehicle_id']} of type {vehicle_data['vehicle_type']}.
Your task is to provide optimal routing recommendations, including multimodal options with buses and trains when appropriate.

CURRENT VEHICLE STATUS:
- Vehicle ID: {vehicle_data['vehicle_id']}
- Vehicle Type: {vehicle_data['vehicle_type']}
- Current Route: {vehicle_data['current_route']}
- Edge History: {', '.join(vehicle_data['edge_history'][-5:]) if vehicle_data['edge_history'] else 'No history yet'}
- Currently Using Multimodal Journey: {vehicle_data['multimodal_journey']}
- Transport Modes Used: {', '.join(vehicle_data['transport_modes']) if vehicle_data['transport_modes'] else 'Private vehicle only'}

TRAFFIC NETWORK STATUS:
"""
        
        # Add information about network edges
        for edge_id, edge_data in network_data.items():
            prompt += f"""
Edge ID: {edge_id}
- Mean Speed: {edge_data['mean_speed']:.2f} m/s
- Occupancy: {edge_data['occupancy']:.2f}
- Vehicle Count: {edge_data['vehicle_count']:.1f}
- Travel Time: {edge_data['travel_time']:.2f} seconds
"""
        
        # Add travel time information if available
        if vehicle_data['travel_times']:
            prompt += "\nVEHICLE TRAVEL TIMES:\n"
            for edge_id, time_data in vehicle_data['travel_times'].items():
                prompt += f"- Edge {edge_id}: {time_data['time']} seconds\n"
        
        # Add public transit information
        prompt += "\nPUBLIC TRANSIT OPTIONS:\n"
        
        if transit_data["nearby_stops"]:
            prompt += "Nearby Transit Stops:\n"
            for stop in transit_data["nearby_stops"]:
                prompt += f"- Stop {stop['id']} on edge {stop['edge_id']}\n"
        else:
            prompt += "No nearby transit stops detected.\n"
        
        if transit_data["relevant_buses"]:
            prompt += "\nAvailable Buses:\n"
            for bus in transit_data["relevant_buses"]:
                prompt += f"- Bus {bus['id']} (Line {bus['line']}), currently on edge {bus['current_edge']}, next stop: {bus['next_stop']}\n"
        
        if transit_data["relevant_trains"]:
            prompt += "\nAvailable Trains:\n"
            for train in transit_data["relevant_trains"]:
                prompt += f"- Train {train['id']} (Line {train['line']}), currently on edge {train['current_edge']}, next stop: {train['next_stop']}\n"
        
        # Request routing recommendations including multimodal options
        prompt += """
Based on the traffic network status and public transit options, please provide routing recommendations for this vehicle.
Analyze the traffic conditions and suggest improvements to the vehicle's journey, including multimodal options with buses and trains when appropriate.

INSTRUCTIONS:
1. Identify congested edges (high travel time, low speed, high occupancy)
2. Consider both private vehicle routing and public transit options
3. Recommend switching to public transit if it would be more efficient
4. Consider the vehicle's type and history in your recommendations
5. Provide a brief explanation for your recommendations

Output your response in JSON format as follows:
```json
{
  "route_adjustments": [
    {
      "edge_to_avoid": "edge_id",
      "alternative_edge": "edge_id",
      "confidence": 0.8
    }
  ],
  "public_transit_recommendations": [
    {
      "type": "bus",
      "id": "bus_id",
      "board_at_stop": "stop_id",
      "alight_at_stop": "stop_id",
      "confidence": 0.9
    }
  ],
  "multimodal_strategy": {
    "use_multimodal": true,
    "transport_modes": ["car", "bus", "train"],
    "transitions": [
      {
        "from_mode": "car",
        "to_mode": "bus",
        "at_location": "stop_id",
        "time_saved_estimate": 300
      }
    ]
  },
  "explanation": "Brief explanation of your recommendations"
}
```
"""
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """Query LLM with the prompt
        
        Args:
            prompt: Prompt string
            
        Returns:
            Response from LLM
        """
        if not self.openai_api_key:
            # Generate a simulated response if no API key
            return self._generate_simulated_llm_response()
        
        try:
            # Send request to OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a traffic navigation assistant for SUMO traffic simulation."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            # Parse the response
            response_data = response.json()
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                print("Unexpected response format from OpenAI API")
                return self._generate_simulated_llm_response()
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._generate_simulated_llm_response()
    
    def _generate_simulated_llm_response(self) -> str:
        """Generate a simulated LLM response when API is unavailable
        
        Returns:
            Simulated response string
        """
        print("  Generating simulated LLM response")
        
        # Create a reasonable simulated response that includes multimodal options
        edge_to_avoid = f"edge_{random.randint(1, 9)}"
        alternative_edge = f"edge_{random.randint(1, 9)}"
        confidence = round(random.uniform(0.6, 0.95), 2)
        
        # Randomly decide whether to recommend public transit
        recommend_transit = random.random() < 0.4
        
        response = {
            "route_adjustments": [
                {
                    "edge_to_avoid": edge_to_avoid,
                    "alternative_edge": alternative_edge,
                    "confidence": confidence
                }
            ],
            "explanation": "Based on traffic conditions, I've identified congestion on the specified edge."
        }
        
        if recommend_transit:
            # Add public transit recommendations
            transit_type = "bus" if random.random() < 0.7 else "train"
            transit_id = f"{transit_type}_{random.randint(0, 2)}"
            board_stop = f"stop_{random.randint(0, 4)}" if transit_type == "bus" else f"station_{random.randint(0, 2)}"
            alight_stop = f"stop_{random.randint(0, 4)}" if transit_type == "bus" else f"station_{random.randint(0, 2)}"
            
            response["multimodal_strategy"] = {
                "use_multimodal": True,
                "transport_modes": ["car", transit_type],
                "transitions": [
                    {
                        "from_mode": "car",
                        "to_mode": transit_type,
                        "at_location": board_stop,
                        "time_saved_estimate": random.randint(100, 500)
                    }
                ]
            }
            
            response["explanation"] += f" I recommend switching to {transit_type} at {board_stop} to save time and avoid traffic congestion."
        else:
            response["public_transit_recommendations"] = []
            response["multimodal_strategy"] = {
                "use_multimodal": False,
                "transport_modes": ["car"],
                "transitions": []
            }
            response["explanation"] += " The private vehicle route is currently optimal, so no public transit recommendations at this time."
        
        return f"```json\n{json.dumps(response, indent=2)}\n```"
    
    def _parse_llm_suggestions(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response and extract route adjustment suggestions
        
        Args:
            llm_response: String response from LLM
            
        Returns:
            Dictionary with route adjustments and multimodal strategy
        """
        try:
            # Extract JSON from markdown code blocks if present
            json_start = llm_response.find("```json")
            if json_start != -1:
                json_start = llm_response.find("\n", json_start) + 1
                json_end = llm_response.find("```", json_start)
                json_str = llm_response[json_start:json_end].strip()
            else:
                # Try to find any JSON object in the response
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    print("  No JSON found in LLM response")
                    return {}
                json_str = llm_response[json_start:json_end].strip()
            
            # Parse the JSON
            recommendations = json.loads(json_str)
            
            print(f"  Received routing recommendations:")
            print(f"  Explanation: {recommendations.get('explanation', 'No explanation provided')}")
            
            # Process route adjustments
            for adjustment in recommendations.get("route_adjustments", []):
                edge_to_avoid = adjustment.get("edge_to_avoid")
                alternative_edge = adjustment.get("alternative_edge")
                confidence = adjustment.get("confidence", 0.0)
                
                print(f"  - Avoid {edge_to_avoid}, use {alternative_edge} instead (confidence: {confidence:.2f})")
            
            # Process public transit recommendations
            for transit_rec in recommendations.get("public_transit_recommendations", []):
                transit_type = transit_rec.get("type")
                transit_id = transit_rec.get("id")
                board_stop = transit_rec.get("board_at_stop")
                alight_stop = transit_rec.get("alight_at_stop")
                confidence = transit_rec.get("confidence", 0.0)
                
                print(f"  - Take {transit_type} {transit_id} from {board_stop} to {alight_stop} (confidence: {confidence:.2f})")
            
            # Process multimodal strategy
            multimodal = recommendations.get("multimodal_strategy", {})
            if multimodal.get("use_multimodal", False):
                self.multimodal_journey = True
                self.transport_modes = multimodal.get("transport_modes", ["car"])
                
                print(f"  - Multimodal journey recommended with modes: {', '.join(self.transport_modes)}")
                
                for transition in multimodal.get("transitions", []):
                    from_mode = transition.get("from_mode")
                    to_mode = transition.get("to_mode")
                    location = transition.get("at_location")
                    time_saved = transition.get("time_saved_estimate", 0)
                    
                    print(f"  - Transit from {from_mode} to {to_mode} at {location} (saves ~{time_saved} seconds)")
            
            return recommendations
            
        except Exception as e:
            print(f"Error parsing LLM suggestions: {e}")
            return {}
